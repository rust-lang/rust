# Appendix A: A tutorial on creating a drop-in replacement for rustc

> **Note:** This is a copy of `@nrc`'s amazing [stupid-stats]. You should find
> a copy of the code on the GitHub repository.
>
> Due to the compiler's constantly evolving nature, the `rustc_driver`
> mechanisms described in this chapter have been replaced by a new
> [`rustc_interface`] crate. See [The Rustc Driver and Interface] for more
> information.

Many tools benefit from being a drop-in replacement for a compiler. By this, I
mean that any user of the tool can use `mytool` in all the ways they would
normally use `rustc` - whether manually compiling a single file or as part of a
complex make project or Cargo build, etc. That could be a lot of work;
rustc, like most compilers, takes a large number of command line arguments which
can affect compilation in complex and interacting ways. Emulating all of this
behaviour in your tool is annoying at best, especically if you are making many
of the same calls into librustc that the compiler is.

The kind of things I have in mind are tools like rustdoc or a future rustfmt.
These want to operate as closely as possible to real compilation, but have
totally different outputs (documentation and formatted source code,
respectively). Another use case is a customised compiler. Say you want to add a
custom code generation phase after macro expansion, then creating a new tool
should be easier than forking the compiler (and keeping it up to date as the
compiler evolves).

I have gradually been trying to improve the API of librustc to make creating a
drop-in tool easier to produce (many others have also helped improve these
interfaces over the same time frame). It is now pretty simple to make a tool
which is as close to rustc as you want it to be. In this tutorial I'll show
how.

Note/warning, everything I talk about in this tutorial is internal API for
rustc. It is all extremely unstable and likely to change often and in
unpredictable ways. Maintaining a tool which uses these APIs will be non-
trivial, although hopefully easier than maintaining one that does similar things
without using them.

This tutorial starts with a very high level view of the rustc compilation
process and of some of the code that drives compilation. Then I'll describe how
that process can be customised. In the final section of the tutorial, I'll go
through an example - stupid-stats - which shows how to build a drop-in tool.


## Overview of the compilation process

Compilation using rustc happens in several phases. We start with parsing, this
includes lexing. The output of this phase is an AST (abstract syntax tree).
There is a single AST for each crate (indeed, the entire compilation process
operates over a single crate). Parsing abstracts away details about individual
files which will all have been read in to the AST in this phase. At this stage
the AST includes all macro uses, attributes will still be present, and nothing
will have been eliminated due to `cfg`s.

The next phase is configuration and macro expansion. This can be thought of as a
function over the AST. The unexpanded AST goes in and an expanded AST comes out.
Macros and syntax extensions are expanded, and `cfg` attributes will cause some
code to disappear. The resulting AST won't have any macros or macro uses left
in.

The code for these first two phases is in [libsyntax](https://github.com/rust-lang/rust/tree/master/src/libsyntax).

After this phase, the compiler allocates ids to each node in the AST
(technically not every node, but most of them). If we are writing out
dependencies, that happens now.

The next big phase is analysis. This is the most complex phase and
uses the bulk of the code in rustc. This includes name resolution, type
checking, borrow checking, type and lifetime inference, trait selection, method
selection, linting, and so forth. Most error detection is done in this phase
(although parse errors are found during parsing). The 'output' of this phase is
a bunch of side tables containing semantic information about the source program.
The analysis code is in [librustc](https://github.com/rust-lang/rust/tree/master/src/librustc)
and a bunch of other crates with the 'librustc_' prefix.

Next is translation, this translates the AST (and all those side tables) into
LLVM IR (intermediate representation). We do this by calling into the LLVM
libraries, rather than actually writing IR directly to a file. The code for
this is in librustc_trans.

The next phase is running the LLVM backend. This runs LLVM's optimisation passes
on the generated IR and then generates machine code. The result is object files.
This phase is all done by LLVM, it is not really part of the rust compiler. The
interface between LLVM and rustc is in [librustc_llvm](https://github.com/rust-lang/rust/tree/master/src/librustc_llvm).

Finally, we link the object files into an executable. Again we outsource this to
other programs and it's not really part of the rust compiler. The interface is
in librustc_back (which also contains some things used primarily during
translation).

> NOTE: `librustc_trans` and `librustc_back` no longer exist, and we don't
> translate AST or HIR directly to LLVM IR anymore.  Instead, see
> [`librustc_codegen_llvm`](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_codegen_llvm/index.html)
> and [`librustc_codegen_utils`](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_codegen_utils/index.html).

All these phases are coordinated by the driver. To see the exact sequence, look
at the `compile_input` function in `librustc_driver`.
The driver handles all the highest level coordination of compilation -
    1. handling command-line arguments
    2. maintaining compilation state (primarily in the `Session`)
    3. calling the appropriate code to run each phase of compilation
    4. handles high level coordination of pretty printing and testing.
To create a drop-in compiler replacement or a compiler replacement,
we leave most of compilation alone and customise the driver using its APIs.

## The driver customisation APIs

There are two primary ways to customise compilation - high level control of the
driver using `CompilerCalls` and controlling each phase of compilation using a
`CompileController`. The former lets you customise handling of command line
arguments etc., the latter lets you stop compilation early or execute code
between phases.


### `CompilerCalls`

`CompilerCalls` is a trait that you implement in your tool. It contains a fairly
ad-hoc set of methods to hook in to the process of processing command line
arguments and driving the compiler. For details, see the comments in
[librustc_driver/lib.rs](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_driver/index.html).
I'll summarise the methods here.

`early_callback` and `late_callback` let you call arbitrary code at different
points - early is after command line arguments have been parsed, but before
anything is done with them; late is pretty much the last thing before
compilation starts, i.e., after all processing of command line arguments, etc.
is done. Currently, you get to choose whether compilation stops or continues at
each point, but you don't get to change anything the driver has done. You can
record some info for later, or perform other actions of your own.

`some_input` and `no_input` give you an opportunity to modify the primary input
to the compiler (usually the input is a file containing the top module for a
crate, but it could also be a string). You could record the input or perform
other actions of your own.

Ignore `parse_pretty`, it is unfortunate and hopefully will get improved. There
is a default implementation, so you can pretend it doesn't exist.

`build_controller` returns a `CompileController` object for more fine-grained
control of compilation, it is described next.

We might add more options in the future.


### `CompilerController`

`CompilerController` is a struct consisting of `PhaseController`s and flags.
Currently, there is only flag, `make_glob_map` which signals whether to produce
a map of glob imports (used by save-analysis and potentially other tools). There
are probably flags in the session that should be moved here.

There is a `PhaseController` for each of the phases described in the above
summary of compilation (and we could add more in the future for finer-grained
control). They are all `after_` a phase because they are checked at the end of a
phase (again, that might change), e.g., `CompilerController::after_parse`
controls what happens immediately after parsing (and before macro expansion).

Each `PhaseController` contains a flag called `stop` which indicates whether
compilation should stop or continue, and a callback to be executed at the point
indicated by the phase. The callback is called whether or not compilation
continues.

Information about the state of compilation is passed to these callbacks in a
`CompileState` object. This contains all the information the compiler has. Note
that this state information is immutable - your callback can only execute code
using the compiler state, it can't modify the state. (If there is demand, we
could change that). The state available to a callback depends on where during
compilation the callback is called. For example, after parsing there is an AST
but no semantic analysis (because the AST has not been analysed yet). After
translation, there is translation info, but no AST or analysis info (since these
have been consumed/forgotten).


## An example - stupid-stats

Our example tool is very simple, it simply collects some simple and not very
useful statistics about a program; it is called stupid-stats. You can find
the (more heavily commented) complete source for the example on [Github](https://github.com/nick29581/stupid-stats/blob/master/src).
To build, just do `cargo build`. To run on a file `foo.rs`, do `cargo run
foo.rs` (assuming you have a Rust program called `foo.rs`. You can also pass any
command line arguments that you would normally pass to rustc). When you run it
you'll see output similar to

```text
In crate: foo,

Found 12 uses of `println!`;
The most common number of arguments is 1 (67% of all functions);
25% of functions have four or more arguments.
```

To make things easier, when we talk about functions, we're excluding methods and
closures.

You can also use the executable as a drop-in replacement for rustc, because
after all, that is the whole point of this exercise. So, however you use rustc
in your makefile setup, you can use `target/stupid` (or whatever executable you
end up with) instead. That might mean setting an environment variable or it
might mean renaming your executable to `rustc` and setting your PATH. Similarly,
if you're using Cargo, you'll need to rename the executable to rustc and set the
PATH. Alternatively, you should be able to use
[multirust](https://github.com/brson/multirust) to get around all the PATH stuff
(although I haven't actually tried that).

(Note that this example prints to stdout. I'm not entirely sure what Cargo does
with stdout from rustc under different circumstances. If you don't see any
output, try inserting a `panic!` after the `println!`s to error out, then Cargo
should dump stupid-stats' stdout to Cargo's stdout).

Let's start with the `main` function for our tool, it is pretty simple:

```rust,ignore
fn main() {
    let args: Vec<_> = std::env::args().collect();
    rustc_driver::run_compiler(&args, &mut StupidCalls::new());
    std::env::set_exit_status(0);
}
```

The first line grabs any command line arguments. The second line calls the
compiler driver with those arguments. The final line sets the exit code for the
program.

The only interesting thing is the `StupidCalls` object we pass to the driver.
This is our implementation of the `CompilerCalls` trait and is what will make
this tool different from rustc.

`StupidCalls` is a mostly empty struct:

```rust,ignore
struct StupidCalls {
    default_calls: RustcDefaultCalls,
}
```

This tool is so simple that it doesn't need to store any data here, but usually
you would. We embed a `RustcDefaultCalls` object to delegate to in our impl when
we want exactly the same behaviour as the Rust compiler. Mostly you don't want
to do that (or at least don't need to) in a tool. However, Cargo calls rustc
with the `--print file-names`, so we delegate in `late_callback` and `no_input`
to keep Cargo happy.

Most of the rest of the impl of `CompilerCalls` is trivial:

```rust,ignore
impl<'a> CompilerCalls<'a> for StupidCalls {
    fn early_callback(&mut self,
                        _: &getopts::Matches,
                        _: &config::Options,
                        _: &diagnostics::registry::Registry,
                        _: ErrorOutputType)
                      -> Compilation {
        Compilation::Continue
    }

    fn late_callback(&mut self,
                     t: &TransCrate,
                     m: &getopts::Matches,
                     s: &Session,
                     c: &CrateStore,
                     i: &Input,
                     odir: &Option<PathBuf>,
                     ofile: &Option<PathBuf>)
                     -> Compilation {
        self.default_calls.late_callback(t, m, s, c, i, odir, ofile);
        Compilation::Continue
    }

    fn some_input(&mut self,
                  input: Input,
                  input_path: Option<Path>)
                  -> (Input, Option<Path>) {
        (input, input_path)
    }

    fn no_input(&mut self,
                m: &getopts::Matches,
                o: &config::Options,
                odir: &Option<Path>,
                ofile: &Option<Path>,
                r: &diagnostics::registry::Registry)
                -> Option<(Input, Option<Path>)> {
        self.default_calls.no_input(m, o, odir, ofile, r);

        // This is not optimal error handling.
        panic!("No input supplied to stupid-stats");
    }

    fn build_controller(&mut self, _: &Session) -> driver::CompileController<'a> {
        ...
    }
}
```

We don't do anything for either of the callbacks, nor do we change the input if
the user supplies it. If they don't, we just `panic!`, this is the simplest way
to handle the error, but not very user-friendly, a real tool would give a
constructive message or perform a default action.

In `build_controller` we construct our `CompileController`. We only want to
parse, and we want to inspect macros before expansion, so we make compilation
stop after the first phase (parsing). The callback after that phase is where the
tool does it's actual work by walking the AST. We do that by creating an AST
visitor and making it walk the AST from the top (the crate root). Once we've
walked the crate, we print the stats we've collected:

```rust,ignore
fn build_controller(&mut self, _: &Session) -> driver::CompileController<'a> {
    // We mostly want to do what rustc does, which is what basic() will return.
    let mut control = driver::CompileController::basic();
    // But we only need the AST, so we can stop compilation after parsing.
    control.after_parse.stop = Compilation::Stop;

    // And when we stop after parsing we'll call this closure.
    // Note that this will give us an AST before macro expansions, which is
    // not usually what you want.
    control.after_parse.callback = box |state| {
        // Which extracts information about the compiled crate...
        let krate = state.krate.unwrap();

        // ...and walks the AST, collecting stats.
        let mut visitor = StupidVisitor::new();
        visit::walk_crate(&mut visitor, krate);

        // And finally prints out the stupid stats that we collected.
        let cratename = match attr::find_crate_name(&krate.attrs[]) {
            Some(name) => name.to_string(),
            None => String::from_str("unknown_crate"),
        };
        println!("In crate: {},\n", cratename);
        println!("Found {} uses of `println!`;", visitor.println_count);

        let (common, common_percent, four_percent) = visitor.compute_arg_stats();
        println!("The most common number of arguments is {} ({:.0}% of all functions);",
                 common, common_percent);
        println!("{:.0}% of functions have four or more arguments.", four_percent);
    };

    control
}
```

That is all it takes to create your own drop-in compiler replacement or custom
compiler! For the sake of completeness I'll go over the rest of the stupid-stats
tool.

```rust
struct StupidVisitor {
    println_count: usize,
    arg_counts: Vec<usize>,
}
```

The `StupidVisitor` struct just keeps track of the number of `println!`s it has
seen and the count for each number of arguments. It implements
`syntax::visit::Visitor` to walk the AST. Mostly we just use the default
methods, these walk the AST taking no action. We override `visit_item` and
`visit_mac` to implement custom behaviour when we walk into items (items include
functions, modules, traits, structs, and so forth, we're only interested in
functions) and macros:

```rust,ignore
impl<'v> visit::Visitor<'v> for StupidVisitor {
    fn visit_item(&mut self, i: &'v ast::Item) {
        match i.node {
            ast::Item_::ItemFn(ref decl, _, _, _, _) => {
                // Record the number of args.
                self.increment_args(decl.inputs.len());
            }
            _ => {}
        }

        // Keep walking.
        visit::walk_item(self, i)
    }

    fn visit_mac(&mut self, mac: &'v ast::Mac) {
        // Find its name and check if it is "println".
        let ast::Mac_::MacInvocTT(ref path, _, _) = mac.node;
        if path_to_string(path) == "println" {
            self.println_count += 1;
        }

        // Keep walking.
        visit::walk_mac(self, mac)
    }
}
```

The `increment_args` method increments the correct count in
`StupidVisitor::arg_counts`. After we're done walking, `compute_arg_stats` does
some pretty basic maths to come up with the stats we want about arguments.


## What next?

These APIs are pretty new and have a long way to go until they're really good.
If there are improvements you'd like to see or things you'd like to be able to
do, let me know in a comment or [GitHub issue](https://github.com/rust-lang/rust/issues).
In particular, it's not clear to me exactly what extra flexibility is required.
If you have an existing tool that would be suited to this setup, please try it
out and let me know if you have problems.

It'd be great to see Rustdoc converted to using these APIs, if that is possible
(although long term, I'd prefer to see Rustdoc run on the output from save-
analysis, rather than doing its own analysis). Other parts of the compiler
(e.g., pretty printing, testing) could be refactored to use these APIs
internally (I already changed save-analysis to use `CompilerController`). I've
been experimenting with a prototype rustfmt which also uses these APIs.

[stupid-stats]: https://github.com/nrc/stupid-stats
[`rustc_interface`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_interface/index.html
[The Rustc Driver and Interface]: ../rustc-driver.html

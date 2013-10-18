% Rust Condition and Error-handling Tutorial

# Introduction

Rust does not provide exception handling[^why-no-exceptions]
in the form most commonly seen in other programming languages such as C++ or Java.
Instead, it provides four mechanisms that work together to handle errors or other rare events.
The four mechanisms are:

  - Options
  - Results
  - Failure
  - Conditions

This tutorial will lead you through use of these mechanisms
in order to understand the trade-offs of each and relationships between them.

# Example program

This tutorial will be based around an example program
that attempts to read lines from a file
consisting of pairs of numbers,
and then print them back out with slightly different formatting.
The input to the program might look like this:

~~~~ {.notrust}
$ cat numbers.txt
1 2
34 56
789 123
45 67
~~~~

For which the intended output looks like this:

~~~~ {.notrust}
$ ./example numbers.txt
0001, 0002
0034, 0056
0789, 0123
0045, 0067
~~~~

An example program that does this task reads like this:

~~~~{.xfail-test}
# #[allow(unused_imports)];
extern mod extra;
use extra::fileinput::FileInput;
use std::int;
# mod FileInput {
#    use std::io::{Reader, BytesReader};
#    static s : &'static [u8] = bytes!("1 2\n\
#                                       34 56\n\
#                                       789 123\n\
#                                       45 67\n\
#                                       ");
#    pub fn from_args() -> @Reader{
#        @BytesReader {
#            bytes: s,
#            pos: @mut 0
#        } as @Reader
#    }
# }

fn main() {
    let pairs = read_int_pairs();
    for &(a,b) in pairs.iter() {
        println!("{:4.4d}, {:4.4d}", a, b);
    }
}


fn read_int_pairs() -> ~[(int,int)] {

    let mut pairs = ~[];

    let fi = FileInput::from_args();
    while ! fi.eof() {

        // 1. Read a line of input.
        let line = fi.read_line();

        // 2. Split the line into fields ("words").
        let fields = line.word_iter().to_owned_vec();

        // 3. Match the vector of fields against a vector pattern.
        match fields {

            // 4. When the line had two fields:
            [a, b] => {

                // 5. Try parsing both fields as ints.
                match (from_str::<int>(a), from_str::<int>(b)) {

                    // 6. If parsing succeeded for both, push both.
                    (Some(a), Some(b)) => pairs.push((a,b)),

                    // 7. Ignore non-int fields.
                    _ => ()
                }
            }

            // 8. Ignore lines that don't have 2 fields.
            _ => ()
        }
    }

    pairs
}
~~~~

This example shows the use of `Option`,
along with some other forms of error-handling (and non-handling).
We will look at these mechanisms
and then modify parts of the example to perform "better" error handling.


# Options

The simplest and most lightweight mechanism in Rust for indicating an error is the type `std::option::Option<T>`.
This type is a general purpose `enum`
for conveying a value of type `T`, represented as `Some(T)`
_or_ the sentinel `None`, to indicate the absence of a `T` value.
For simple APIs, it may be sufficient to encode errors as `Option<T>`,
returning `Some(T)` on success and `None` on error.
In the example program, the call to `from_str::<int>` returns `Option<int>`
with the understanding that "all parse errors" result in `None`.
The resulting `Option<int>` values are matched against the pattern `(Some(a), Some(b))`
in steps 5 and 6 in the example program,
to handle the case in which both fields were parsed successfully.

Using `Option` as in this API has some advantages:

  - Simple API, users can read it and guess how it works.
  - Very efficient, only an extra `enum` tag on return values.
  - Caller has flexibility in handling or propagating errors.
  - Caller is forced to acknowledge existence of possible-error before using value.

However, it has serious disadvantages too:

  - Verbose, requires matching results or calling `Option::unwrap` everywhere.
  - Infects caller: if caller doesn't know how to handle the error, must propagate (or force).
  - Temptation to do just that: force the `Some(T)` case by blindly calling `unwrap`,
    which hides the error from the API without providing any way to make the program robust against the error.
  - Collapses all errors into one:
    - Caller can't handle different errors differently.
    - Caller can't even report a very precise error message

Note that in order to keep the example code reasonably compact,
several unwanted cases are silently ignored:
lines that do not contain two fields, as well as fields that do not parse as ints.
To propagate these cases to the caller using `Option` would require even more verbose code.


# Results

Before getting into _trapping_ the error,
we will look at a slight refinement on the `Option` type above.
This second mechanism for indicating an error is called a `Result`.
The type `std::result::Result<T,E>` is another simple `enum` type with two forms, `Ok(T)` and `Err(E)`.
The `Result` type is not substantially different from the `Option` type in terms of its ergonomics.
Its main advantage is that the error constructor `Err(E)` can convey _more detail_ about the error.
For example, the `from_str` API could be reformed
to return a `Result` carrying an informative description of a parse error,
like this:

~~~~ {.ignore}
enum IntParseErr {
     EmptyInput,
     Overflow,
     BadChar(char)
}

fn from_str(&str) -> Result<int,IntParseErr> {
  // ...
}
~~~~

This would give the caller more information for both handling and reporting the error,
but would otherwise retain the verbosity problems of using `Option`.
In particular, it would still be necessary for the caller to return a further `Result` to _its_ caller if it did not want to handle the error.
Manually propagating result values this way can be attractive in certain circumstances
-- for example when processing must halt on the very first error, or backtrack --
but as we will see later, many cases have simpler options available.

# Failure

The third and arguably easiest mechanism for handling errors is called "failure".
In fact it was hinted at earlier by suggesting that one can choose to propagate `Option` or `Result` types _or "force" them_.
"Forcing" them, in this case, means calling a method like `Option<T>::unwrap`,
which contains the following code:

~~~~ {.ignore}
pub fn unwrap(self) -> T {
    match self {
      Some(x) => return x,
      None => fail!("option::unwrap `None`")
    }
}
~~~~

That is, it returns `T` when `self` is `Some(T)`, and  _fails_ when `self` is `None`.

Every Rust task can _fail_, either indirectly due to a kill signal or other asynchronous event,
or directly by failing an `assert!` or calling the `fail!` macro.
Failure is an _unrecoverable event_ at the task level:
it causes the task to halt normal execution and unwind its control stack,
freeing all task-local resources (the local heap as well as any task-owned values from the global heap)
and running destructors (the `drop` method of the `Drop` trait)
as frames are unwound and heap values destroyed.
A failing task is not permitted to "catch" the unwinding during failure and recover,
it is only allowed to clean up and exit.

Failure has advantages:

  - Simple and non-verbose. Suitable for programs that can't reasonably continue past an error anyways.
  - _All_ errors (except memory-safety errors) can be uniformly trapped in a supervisory task outside the failing task.
    For a large program to be robust against a variety of errors,
    often some form of task-level partitioning to contain pervasive errors (arithmetic overflow, division by zero,
    logic bugs) is necessary anyways.

As well as obvious disadvantages:

  - A blunt instrument, terminates the containing task entirely.

Recall that in the first two approaches to error handling,
the example program was only handling success cases, and ignoring error cases.
That is, if the input is changed to contain a malformed line:

~~~~ {.notrust}
$ cat bad.txt
1 2
34 56
ostrich
789 123
45 67
~~~~

Then the program would give the same output as if there was no error:

~~~~ {.notrust}
$ ./example bad.txt
0001, 0002
0034, 0056
0789, 0123
0045, 0067
~~~~

If the example is rewritten to use failure, these error cases can be trapped.
In this rewriting, failures are trapped by placing the I/O logic in a sub-task,
and trapping its exit status using `task::try`:

~~~~ {.xfail-test}
# #[allowed(unused_imports)];
extern mod extra;
use extra::fileinput::FileInput;
use std::int;
use std::task;
# mod FileInput {
#    use std::io::{Reader, BytesReader};
#    static s : &'static [u8] = bytes!("1 2\n\
#                                       34 56\n\
#                                       ostrich\n\
#                                       789 123\n\
#                                       45 67\n\
#                                       ");
#    pub fn from_args() -> @Reader{
#        @BytesReader {
#            bytes: s,
#            pos: @mut 0
#        } as @Reader
#    }
# }

fn main() {

    // Isolate failure within a subtask.
    let result = do task::try {

        // The protected logic.
        let pairs = read_int_pairs();
        for &(a,b) in pairs.iter() {
            println!("{:4.4d}, {:4.4d}", a, b);
        }

    };
    if result.is_err() {
            println("parsing failed");
    }
}

fn read_int_pairs() -> ~[(int,int)] {
    let mut pairs = ~[];
    let fi = FileInput::from_args();
    while ! fi.eof() {
        let line = fi.read_line();
        let fields = line.word_iter().to_owned_vec();
        match fields {
            [a, b] => pairs.push((from_str::<int>(a).unwrap(),
                                  from_str::<int>(b).unwrap())),

            // Explicitly fail on malformed lines.
            _ => fail!()
        }
    }
    pairs
}
~~~~

With these changes in place, running the program on malformed input gives a different answer:

~~~~ {.notrust}
$ ./example bad.txt
rust: task failed at 'explicit failure', ./example.rs:44
parsing failed
~~~~

Note that while failure unwinds the sub-task performing I/O in `read_int_pairs`,
control returns to `main` and can easily continue uninterrupted.
In this case, control simply prints out `parsing failed` and then exits `main` (successfully).
Failure of a (sub-)task is analogous to calling `exit(1)` or `abort()` in a unix C program:
all the state of a sub-task is cleanly discarded on exit,
and a supervisor task can take appropriate action
without worrying about its own state having been corrupted.


# Conditions

The final mechanism for handling errors is called a "condition".
Conditions are less blunt than failure, and less cumbersome than the `Option` or `Result` types;
indeed they are designed to strike just the right balance between the two.
Conditions require some care to use effectively, but give maximum flexibility with minimum verbosity.
While conditions use exception-like terminology ("trap", "raise") they are significantly different:

  - Like exceptions and failure, conditions separate the site at which the error is raised from the site where it is trapped.
  - Unlike exceptions and unlike failure, when a condition is raised and trapped, _no unwinding occurs_.
  - A successfully trapped condition causes execution to continue _at the site of the error_, as though no error occurred.

Conditions are declared with the `condition!` macro.
Each condition has a name, an input type and an output type, much like a function.
In fact, conditions are implemented as dynamically-scoped functions held in task local storage.

The `condition!` macro declares a module with the name of the condition;
the module contains a single static value called `cond`, of type `std::condition::Condition`.
The `cond` value within the module is the rendezvous point
between the site of error and the site that handles the error.
It has two methods of interest: `raise` and `trap`.

The `raise` method maps a value of the condition's input type to its output type.
The input type should therefore convey all relevant information to the condition handler.
The output type should convey all relevant information _for continuing execution at the site of error_.
When the error site raises a condition handler,
the `Condition::raise` method searches task-local storage (TLS) for the innermost installed _handler_,
and if any such handler is found, calls it with the provided input value.
If no handler is found, `Condition::raise` will fail the task with an appropriate error message.

Rewriting the example to use a condition in place of ignoring malformed lines makes it slightly longer,
but similarly clear as the version that used `fail!` in the logic where the error occurs:

~~~~ {.xfail-test}
# #[allow(unused_imports)];
extern mod extra;
use extra::fileinput::FileInput;
use std::int;
# mod FileInput {
#    use std::io::{Reader, BytesReader};
#    static s : &'static [u8] = bytes!("1 2\n\
#                                       34 56\n\
#                                       ostrich\n\
#                                       789 123\n\
#                                       45 67\n\
#                                       ");
#    pub fn from_args() -> @Reader{
#        @BytesReader {
#            bytes: s,
#            pos: @mut 0
#        } as @Reader
#    }
# }

// Introduce a new condition.
condition! {
    pub malformed_line : ~str -> (int,int);
}

fn main() {
    let pairs = read_int_pairs();
    for &(a,b) in pairs.iter() {
        println!("{:4.4d}, {:4.4d}", a, b);
    }
}

fn read_int_pairs() -> ~[(int,int)] {
    let mut pairs = ~[];
    let fi = FileInput::from_args();
    while ! fi.eof() {
        let line = fi.read_line();
        let fields = line.word_iter().to_owned_vec();
        match fields {
            [a, b] => pairs.push((from_str::<int>(a).unwrap(),
                                  from_str::<int>(b).unwrap())),

            // On malformed lines, call the condition handler and
            // push whatever the condition handler returns.
            _ => pairs.push(malformed_line::cond.raise(line.clone()))
        }
    }
    pairs
}
~~~~

When this is run on malformed input, it still fails,
but with a slightly different failure message than before:

~~~~ {.notrust}
$ ./example bad.txt
rust: task failed at 'Unhandled condition: malformed_line: ~"ostrich"', .../libstd/condition.rs:43
~~~~

While this superficially resembles the trapped `fail!` call before,
it is only because the example did not install a handler for the condition.
The different failure message is indicating, among other things,
that the condition-handling system is being invoked and failing
only due to the absence of a _handler_ that traps the condition.

# Trapping a condition

To trap a condition, use `Condition::trap` in some caller of the site that calls `Condition::raise`.
For example, this version of the program traps the `malformed_line` condition
and replaces bad input lines with the pair `(-1,-1)`:

~~~~{.xfail-test}
# #[allow(unused_imports)];
extern mod extra;
use extra::fileinput::FileInput;
use std::int;
# mod FileInput {
#    use std::io::{Reader, BytesReader};
#    static s : &'static [u8] = bytes!("1 2\n\
#                                       34 56\n\
#                                       ostrich\n\
#                                       789 123\n\
#                                       45 67\n\
#                                       ");
#    pub fn from_args() -> @Reader{
#        @BytesReader {
#            bytes: s,
#            pos: @mut 0
#        } as @Reader
#    }
# }

condition! {
    pub malformed_line : ~str -> (int,int);
}

fn main() {
    // Trap the condition:
    do malformed_line::cond.trap(|_| (-1,-1)).inside {

        // The protected logic.
        let pairs = read_int_pairs();
        for &(a,b) in pairs.iter() {
                println!("{:4.4d}, {:4.4d}", a, b);
        }

    }
}

fn read_int_pairs() -> ~[(int,int)] {
    let mut pairs = ~[];
    let fi = FileInput::from_args();
    while ! fi.eof() {
        let line = fi.read_line();
        let fields = line.word_iter().to_owned_vec();
        match fields {
            [a, b] => pairs.push((from_str::<int>(a).unwrap(),
                                  from_str::<int>(b).unwrap())),
            _ => pairs.push(malformed_line::cond.raise(line.clone()))
        }
    }
    pairs
}
~~~~

Note that the remainder of the program is _unchanged_ with this trap in place;
only the caller that installs the trap changed.
Yet when the condition-trapping variant runs on the malformed input,
it continues execution past the malformed line, substituting the handler's return value.

~~~~ {.notrust}
$ ./example bad.txt
0001, 0002
0034, 0056
-0001, -0001
0789, 0123
0045, 0067
~~~~

# Refining a condition

As you work with a condition, you may find that the original set of options you present for recovery is insufficient.
This is no different than any other issue of API design:
a condition handler is an API for recovering from the condition, and sometimes APIs need to be enriched.
In the example program, the first form of the `malformed_line` API implicitly assumes that recovery involves a substitute value.
This assumption may not be correct; some callers may wish to skip malformed lines, for example.
Changing the condition's return type from `(int,int)` to `Option<(int,int)>` will suffice to support this type of recovery:

~~~~{.xfail-test}
# #[allow(unused_imports)];
extern mod extra;
use extra::fileinput::FileInput;
use std::int;
# mod FileInput {
#    use std::io::{Reader, BytesReader};
#    static s : &'static [u8] = bytes!("1 2\n\
#                                       34 56\n\
#                                       ostrich\n\
#                                       789 123\n\
#                                       45 67\n\
#                                       ");
#    pub fn from_args() -> @Reader{
#        @BytesReader {
#            bytes: s,
#            pos: @mut 0
#        } as @Reader
#    }
# }

// Modify the condition signature to return an Option.
condition! {
    pub malformed_line : ~str -> Option<(int,int)>;
}

fn main() {
    // Trap the condition and return `None`
    do malformed_line::cond.trap(|_| None).inside {

        // The protected logic.
        let pairs = read_int_pairs();
        for &(a,b) in pairs.iter() {
            println!("{:4.4d}, {:4.4d}", a, b);
        }

    }
}

fn read_int_pairs() -> ~[(int,int)] {
    let mut pairs = ~[];
    let fi = FileInput::from_args();
    while ! fi.eof() {
        let line = fi.read_line();
        let fields = line.word_iter().to_owned_vec();
        match fields {
            [a, b] => pairs.push((from_str::<int>(a).unwrap(),
                                  from_str::<int>(b).unwrap())),

            // On malformed lines, call the condition handler and
            // either ignore the line (if the handler returns `None`)
            // or push any `Some(pair)` value returned instead.
            _ => {
                match malformed_line::cond.raise(line.clone()) {
                    Some(pair) => pairs.push(pair),
                    None => ()
                }
            }
        }
    }
    pairs
}
~~~~

Again, note that the remainder of the program is _unchanged_,
in particular the signature of `read_int_pairs` is unchanged,
even though the innermost part of its reading-loop has a new way of handling a malformed line.
When the example is run with the `None` trap in place,
the line is ignored as it was in the first example,
but the choice of whether to ignore or use a substitute value has been moved to some caller,
possibly a distant caller.

~~~~ {.notrust}
$ ./example bad.txt
0001, 0002
0034, 0056
0789, 0123
0045, 0067
~~~~

# Further refining a condition

Like with any API, the process of refining argument and return types of a condition will continue,
until all relevant combinations encountered in practice are encoded.
In the example, suppose a third possible recovery form arose: reusing the previous value read.
This can be encoded in the handler API by introducing a helper type: `enum MalformedLineFix`.

~~~~{.xfail-test}
# #[allow(unused_imports)];
extern mod extra;
use extra::fileinput::FileInput;
use std::int;
# mod FileInput {
#    use std::io::{Reader, BytesReader};
#    static s : &'static [u8] = bytes!("1 2\n\
#                                       34 56\n\
#                                       ostrich\n\
#                                       789 123\n\
#                                       45 67\n\
#                                       ");
#    pub fn from_args() -> @Reader{
#        @BytesReader {
#            bytes: s,
#            pos: @mut 0
#        } as @Reader
#    }
# }

// Introduce a new enum to convey condition-handling strategy to error site.
pub enum MalformedLineFix {
     UsePair(int,int),
     IgnoreLine,
     UsePreviousLine
}

// Modify the condition signature to return the new enum.
// Note: a condition introduces a new module, so the enum must be
// named with the `super::` prefix to access it.
condition! {
    pub malformed_line : ~str -> super::MalformedLineFix;
}

fn main() {
    // Trap the condition and return `UsePreviousLine`
    do malformed_line::cond.trap(|_| UsePreviousLine).inside {

        // The protected logic.
        let pairs = read_int_pairs();
        for &(a,b) in pairs.iter() {
            println!("{:4.4d}, {:4.4d}", a, b);
        }

    }
}

fn read_int_pairs() -> ~[(int,int)] {
    let mut pairs = ~[];
    let fi = FileInput::from_args();
    while ! fi.eof() {
        let line = fi.read_line();
        let fields = line.word_iter().to_owned_vec();
        match fields {
            [a, b] => pairs.push((from_str::<int>(a).unwrap(),
                                  from_str::<int>(b).unwrap())),

            // On malformed lines, call the condition handler and
            // take action appropriate to the enum value returned.
            _ => {
                match malformed_line::cond.raise(line.clone()) {
                    UsePair(a,b) => pairs.push((a,b)),
                    IgnoreLine => (),
                    UsePreviousLine => {
                        let prev = pairs[pairs.len() - 1];
                        pairs.push(prev)
                    }
                }
            }
        }
    }
    pairs
}
~~~~

Running the example with `UsePreviousLine` as the fix code returned from the handler
gives the expected result:

~~~~ {.notrust}
$ ./example bad.txt
0001, 0002
0034, 0056
0034, 0056
0789, 0123
0045, 0067
~~~~

At this point the example has a rich variety of recovery options,
none of which is visible to casual users of the `read_int_pairs` function.
This is intentional: part of the purpose of using a condition
is to free intermediate callers from the burden of having to write repetitive error-propagation logic,
and/or having to change function call and return types as error-handling strategies are refined.

# Multiple conditions, intermediate callers

So far the function trapping the condition and the function raising it have been immediately adjacent in the call stack.
That is, the caller traps and its immediate callee raises.
In most programs, the function that traps may be separated by very many function calls from the function that raises.
Again, this is part of the point of using conditions:
to support that separation without having to thread multiple error values and recovery strategies all the way through the program's main logic.

Careful readers will notice that there is a remaining failure mode in the example program: the call to `.unwrap()` when parsing each integer.
For example, when presented with a file that has the correct number of fields on a line,
but a non-numeric value in one of them, such as this:

~~~~ {.notrust}
$ cat bad.txt
1 2
34 56
7 marmot
789 123
45 67
~~~~


Then the program fails once more:

~~~~ {.notrust}
$ ./example bad.txt
task <unnamed> failed at 'called `Option::unwrap()` on a `None` value', .../libstd/option.rs:314
~~~~

To make the program robust -- or at least flexible -- in the face of this potential failure,
a second condition and a helper function will suffice:

~~~~{.xfail-test}
# #[allow(unused_imports)];
extern mod extra;
use extra::fileinput::FileInput;
use std::int;
# mod FileInput {
#    use std::io::{Reader, BytesReader};
#    static s : &'static [u8] = bytes!("1 2\n\
#                                       34 56\n\
#                                       7 marmot\n\
#                                       789 123\n\
#                                       45 67\n\
#                                       ");
#    pub fn from_args() -> @Reader{
#        @BytesReader {
#            bytes: s,
#            pos: @mut 0
#        } as @Reader
#    }
# }

pub enum MalformedLineFix {
     UsePair(int,int),
     IgnoreLine,
     UsePreviousLine
}

condition! {
    pub malformed_line : ~str -> ::MalformedLineFix;
}

// Introduce a second condition.
condition! {
    pub malformed_int : ~str -> int;
}

fn main() {
    // Trap the `malformed_int` condition and return -1
    do malformed_int::cond.trap(|_| -1).inside {

        // Trap the `malformed_line` condition and return `UsePreviousLine`
        do malformed_line::cond.trap(|_| UsePreviousLine).inside {

            // The protected logic.
            let pairs = read_int_pairs();
            for &(a,b) in pairs.iter() {
                println!("{:4.4d}, {:4.4d}", a, b);
            }

        }
    }
}

// Parse an int; if parsing fails, call the condition handler and
// return whatever it returns.
fn parse_int(x: &str) -> int {
    match from_str::<int>(x) {
        Some(v) => v,
        None => malformed_int::cond.raise(x.to_owned())
    }
}

fn read_int_pairs() -> ~[(int,int)] {
    let mut pairs = ~[];
    let fi = FileInput::from_args();
    while ! fi.eof() {
        let line = fi.read_line();
        let fields = line.word_iter().to_owned_vec();
        match fields {

            // Delegate parsing ints to helper function that will
            // handle parse errors by calling `malformed_int`.
            [a, b] => pairs.push((parse_int(a), parse_int(b))),

            _ => {
                match malformed_line::cond.raise(line.clone()) {
                    UsePair(a,b) => pairs.push((a,b)),
                    IgnoreLine => (),
                    UsePreviousLine => {
                        let prev = pairs[pairs.len() - 1];
                        pairs.push(prev)
                    }
                }
            }
        }
    }
    pairs
}
~~~~

Again, note that `read_int_pairs` has not changed signature,
nor has any of the machinery for trapping or raising `malformed_line`,
but now the program can handle the "right number of fields, non-integral field" form of bad input:

~~~~ {.notrust}
$ ./example bad.txt
0001, 0002
0034, 0056
0007, -0001
0789, 0123
0045, 0067
~~~~

There are three other things to note in this variant of the example program:

  - It traps multiple conditions simultaneously,
    nesting the protected logic of one `trap` call inside the other.

  - There is a function in between the `trap` site and `raise` site for the `malformed_int` condition.
    There could be any number of calls between them:
    so long as the `raise` occurs within a callee (of any depth) of the logic protected by the `trap` call,
    it will invoke the handler.

  - This variant insulates callers from a design choice in the library:
    the `from_str` function was designed to return an `Option<int>`,
    but this program insulates callers from that choice,
    routing all `None` values that arise from parsing integers in this file into the condition.


# When to use which technique

This tutorial explored several techniques for handling errors.
Each is appropriate to different circumstances:

  - If an error may be extremely frequent, expected, and very likely dealt with by an immediate caller,
    then returning an `Option` or `Result` type is best. These types force the caller to handle the error,
    and incur the lowest speed overhead, usually only returning one extra word to tag the return value.
    Between `Option` and `Result`: use an `Option` when there is only one kind of error,
    otherwise make an `enum FooErr` to represent the possible error codes and use `Result<T,FooErr>`.

  - If an error can reasonably be handled at the site it occurs by one of a few strategies -- possibly including failure --
    and it is not clear which strategy a caller would want to use, a condition is best.
    For many errors, the only reasonable "non-stop" recovery strategies are to retry some number of times,
    create or substitute an empty or sentinel value, ignore the error, or fail.

  - If an error cannot reasonably be handled at the site it occurs,
    and the only reasonable response is to abandon a large set of operations in progress,
    then directly failing is best.

Note that an unhandled condition will cause failure (along with a more-informative-than-usual message),
so if there is any possibility that a caller might wish to "ignore and keep going",
it is usually harmless to use a condition in place of a direct call to `fail!()`.


[^why-no-exceptions]: Exceptions in languages like C++ and Java permit unwinding, like Rust's failure system,
but with the option to halt unwinding partway through the process and continue execution.
This behavior unfortunately means that the _heap_ may be left in an inconsistent but accessible state,
if an exception is thrown part way through the process of initializing or modifying memory.
To compensate for this risk, correct C++ and Java code must program in an extremely elaborate and difficult "exception-safe" style
-- effectively transactional style against heap structures --
or else risk introducing silent and very difficult-to-debug errors due to control resuming in a corrupted heap after a caught exception.
These errors are frequently memory-safety errors, which Rust strives to eliminate,
and so Rust unwinding is unrecoverable within a single task:
once unwinding starts, the entire local heap of a task is destroyed and the task is terminated.

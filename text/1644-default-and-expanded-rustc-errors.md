- Feature Name: `default_and_expanded_errors_for_rustc`
- Start Date: 2016-06-07
- RFC PR: [rust-lang/rfcs#1644](https://github.com/rust-lang/rfcs/pull/1644)
- Rust Issue: [rust-lang/rust#34826](https://github.com/rust-lang/rust/issues/34826)
              [rust-lang/rust#34827](https://github.com/rust-lang/rust/issues/34827)

# Summary
This RFC proposes an update to error reporting in rustc. Its focus is to change the format of Rust
error messages and improve --explain capabilities to focus on the user's code. The end goal is for
errors and explain text to be more readable, more friendly to new users, while still helping Rust
coders fix bugs as quickly as possible. We expect to follow this RFC with a supplemental RFC that
provides a writing style guide for error messages and explain text with a focus on readability and
education.

# Motivation

## Default error format

Rust offers a unique value proposition in the landscape of languages in part by codifying concepts
like ownership and borrowing. Because these concepts are unique to Rust, it's critical that the
learning curve be as smooth as possible. And one of the most important tools for lowering the
learning curve is providing excellent errors that serve to make the concepts less intimidating,
and to help 'tell the story' about what those concepts mean in the context of the programmer's code.

[as text]
```
src/test/compile-fail/borrowck/borrowck-borrow-from-owned-ptr.rs:29:22: 29:30 error: cannot borrow `foo.bar1` as mutable more than once at a time [E0499]
src/test/compile-fail/borrowck/borrowck-borrow-from-owned-ptr.rs:29     let _bar2 = &mut foo.bar1;
                                                                                         ^~~~~~~~
src/test/compile-fail/borrowck/borrowck-borrow-from-owned-ptr.rs:29:22: 29:30 help: run `rustc --explain E0499` to see a detailed explanation
src/test/compile-fail/borrowck/borrowck-borrow-from-owned-ptr.rs:28:21: 28:29 note: previous borrow of `foo.bar1` occurs here; the mutable borrow prevents subsequent moves, borrows, or modification of `foo.bar1` until the borrow ends
src/test/compile-fail/borrowck/borrowck-borrow-from-owned-ptr.rs:28     let bar1 = &mut foo.bar1;
                                                                                        ^~~~~~~~
src/test/compile-fail/borrowck/borrowck-borrow-from-owned-ptr.rs:31:2: 31:2 note: previous borrow ends here
src/test/compile-fail/borrowck/borrowck-borrow-from-owned-ptr.rs:26 fn borrow_same_field_twice_mut_mut() {
src/test/compile-fail/borrowck/borrowck-borrow-from-owned-ptr.rs:27     let mut foo = make_foo();
src/test/compile-fail/borrowck/borrowck-borrow-from-owned-ptr.rs:28     let bar1 = &mut foo.bar1;
src/test/compile-fail/borrowck/borrowck-borrow-from-owned-ptr.rs:29     let _bar2 = &mut foo.bar1;
src/test/compile-fail/borrowck/borrowck-borrow-from-owned-ptr.rs:30     *bar1;
src/test/compile-fail/borrowck/borrowck-borrow-from-owned-ptr.rs:31 }
                                                                    ^
```

[as image]
![Image of new error flow](http://www.jonathanturner.org/images/old_errors_3.png)

*Example of a borrow check error in the current compiler*

Though a lot of time has been spent on the current error messages, they have a couple flaws which
make them difficult to use. Specifically, the current error format:

* Repeats the file position on the left-hand side. This offers no additional information, but
instead makes the error harder to read.
* Prints messages about lines often out of order. This makes it difficult for the developer to
glance at the error and recognize why the error is occuring
* Lacks a clear visual break between errors. As more errors occur it becomes more difficult to tell
them apart.
* Uses technical terminology that is difficult for new users who may be unfamiliar with compiler
terminology or terminology specific to Rust.

This RFC details a redesign of errors to focus more on the source the programmer wrote. This format
addresses the above concerns by eliminating clutter, following a more natural order for help
messages, and pointing the user to both "what" the error is and "why" the error is occurring by
using color-coded labels. Below you can see the same error again, this time using the proposed
format:

[as text]
```
error[E0499]: cannot borrow `foo.bar1` as mutable more than once at a time
  --> src/test/compile-fail/borrowck/borrowck-borrow-from-owned-ptr.rs:29:22
   |
28 |      let bar1 = &mut foo.bar1;
   |                      -------- first mutable borrow occurs here
29 |      let _bar2 = &mut foo.bar1;
   |                       ^^^^^^^^ second mutable borrow occurs here
30 |      *bar1;
31 |  }
   |  - first borrow ends here
```

[as image]

<a href="http://www.jonathanturner.org/images/new_errors_3.png"><img src="http://www.jonathanturner.org/images/new_errors_3.png" height="200" width="750" ></a>

*Example of the same borrow check error in the proposed format*

## Expanded error format (revised --explain)

Languages like Elm have shown how effective an educational tool error messages can be if the
explanations like our --explain text are mixed with the user's code. As mentioned earlier, it's
crucial for Rust to be easy-to-use, especially since it introduces a fair number of concepts that
may be unfamiliar to the user. Even experienced users may need to use --explain text from time to
time when they encounter unfamiliar messages.

While we have --explain text today, it uses generic examples that require the user to mentally
translate the given example into what works for their specific situation.

```
You tried to move out of a value which was borrowed. Erroneous code example:

use std::cell::RefCell;

struct TheDarkKnight;

impl TheDarkKnight {
    fn nothing_is_true(self) {}
}
...
```

*Example of the current --explain (showing E0507)*

To help users, this RFC proposes a new `--explain errors`. This new mode is more textual error
reporting mode that gives additional explanation to help better understand compiler messages. The
end result is a richer, on-demand error reporting style.

```
error: cannot move out of borrowed content
   --> /Users/jturner/Source/errors/borrowck-move-out-of-vec-tail.rs:30:17

I’m trying to track the ownership of the contents of `tail`, which is borrowed, through this match
statement:

29  |              match tail {

In this match, you use an expression of the form [...]. When you do this, it’s like you are opening
up the `tail` value and taking out its contents. Because `tail` is borrowed, you can’t safely move
the contents.

30  |                  [Foo { string: aa },
    |                                 ^^ cannot move out of borrowed content

You can avoid moving the contents out by working with each part using a reference rather than a
move. A naive fix might look this:

30  |                  [Foo { string: ref aa },

```

# Detailed design

The RFC is separated into two parts: the format of error messages and the format of expanded error
messages (using `--explain errors`).

## Format of error messages

The proposal is a lighter error format focused on the code the user wrote. Messages that help
understand why an error occurred appear as labels on the source. The goals of this new format are
to:

* Create something that's visually easy to parse
* Remove noise/unnecessary information
* Present information in a way that works well for new developers, post-onboarding, and experienced
developers without special configuration
* Draw inspiration from Elm as well as Dybuk and other systems that have already improved on the
kind of errors that Rust has.

In order to accomplish this, the proposed design needs to satisfy a number of constraints to make
the result maximally flexible across various terminals:

* Multiple errors beside each other should be clearly separate and not muddled together.
* Each error message should draw the eye to where the error occurs with sufficient context to
understand why the error occurs.
* Each error should have a "header" section that is visually distinct from the code section.
* Code should visually stand out from text and other error messages. This allows the developer to
immediately recognize their code.
* Error messages should be just as readable when not using colors (eg for users of black-and-white
terminals, color-impaired readers, weird color schemes that we can't predict, or just people that
turn colors off)
* Be careful using “ascii art” and avoid unicode. Instead look for ways to show the information
concisely that will work across the broadest number of terminals. We expect IDEs to possibly allow
for a more graphical error in the future.
* Where possible, use labels on the source itself rather than sentence "notes" at the end.
* Keep filename:line easy to spot for people who use editors that let them click on errors

### Header

```
error[E0499]: cannot borrow `foo.bar1` as mutable more than once at a time
  --> src/test/compile-fail/borrowck/borrowck-borrow-from-owned-ptr.rs:29:22
```

The header still serves the original purpose of knowing: a) if it's a warning or error, b) the text
of the warning/error, and c) the location of this warning/error. We keep the error code, now a part
of the error indicator, as a way to help improve search results.

### Line number column

```
   |
28 |
   |
29 |
   |
30 |
31 |
   |
```

The line number column lets you know where the error is occurring in the file. Because we only show
lines that are of interest for the given error/warning, we elide lines if they are not annotated as
part of the message (we currently use the heuristic to elide after one un-annotated line).

Inspired by Dybuk and Elm, the line numbers are separated with a 'wall', a separator formed from
pipe('|') characters, to clearly distinguish what is a line number from what is source at a glance.

As the wall also forms a way to visually separate distinct errors, we propose extending this concept
to also support span-less notes and hints. For example:

```
92 |         config.target_dir(&pkg)
   |                           ^^^^ expected `core::workspace::Workspace`, found `core::package::Package`
   = note: expected type `&core::workspace::Workspace<'_>`
   = note:    found type `&core::package::Package`
```
### Source area

```
      let bar1 = &mut foo.bar1;
                      -------- first mutable borrow occurs here
      let _bar2 = &mut foo.bar1;
                       ^^^^^^^^ second mutable borrow occurs here
      *bar1;
  }
  - first borrow ends here
```

The source area shows the related source code for the error/warning. The source is laid out in the
order it appears in the source file, giving the user a way to map the message against the source
they wrote.

Key parts of the code are labeled with messages to help the user understand the message.

The primary label is the label associated with the main warning/error. It explains the **what** of
the compiler message. By reading it, the user can begin to understand what the root cause of the
error or warning is. This label is colored to match the level of the message (yellow for warning,
red for error) and uses the ^^^ underline.

Secondary labels help to understand the error and use blue text and --- underline. These labels
explain the **why** of the compiler message. You can see one such example in the above message
where the secondary labels explain that there is already another borrow going on. In another
example, we see another way that primary and secondary work together to tell the whole story for
why the error occurred.

Taken together, primary and secondary labels create a 'flow' to the message. Flow in the message
lets the user glance at the colored labels and quickly form an educated guess as to how to correctly
update their code.

Note: We'll talk more about additional style guidance for wording to help create flow in the
subsequent style RFC.

## Expanded error messages

Currently, --explain text focuses on the error code. You invoke the compiler with --explain
<error code> and receive a verbose description of what causes errors of that number. The resulting
message can be helpful, but it uses generic sample code which makes it feel less connected to the
user's code.

We propose adding a new `--explain errors`. By passing this to the compiler (or to cargo), the
compiler will switch to an expanded error form which incorporates the same source and label
information the user saw in the default message with more explanation text.

```
error: cannot move out of borrowed content
   --> /Users/jturner/Source/errors/borrowck-move-out-of-vec-tail.rs:30:17

I’m trying to track the ownership of the contents of `tail`, which is borrowed, through this match
statement:

29  |              match tail {

In this match, you use an expression of the form [...]. When you do this, it’s like you are opening
up the `tail` value and taking out its contents. Because `tail` is borrowed, you can’t safely move
the contents.

30  |                  [Foo { string: aa },
    |                                 ^^ cannot move out of borrowed content

You can avoid moving the contents out by working with each part using a reference rather than a
move. A naive fix might look this:

30  |                  [Foo { string: ref aa },
```

*Example of an expanded error message*

The expanded error message effectively becomes a template. The text of the template is the
educational text that is explaining the message more more detail. The template is then populated
using the source lines, labels, and spans from the same compiler message that's printed in the
default mode. This lets the message writer call out each label or span as appropriate in the
expanded text.

It's possible to also add additional labels that aren't necessarily shown in the default error mode
but would be available in the expanded error format. This gives the explain text writer maximal
flexibility without impacting the readability of the default message. I'm currently prototyping an
implementation of how this templating could work in practice.

## Tying it together

Lastly, we propose that the final error message:

```
error: aborting due to 2 previous errors
```

Be changed to notify users of this ability:

```
note: compile failed due to 2 errors. You can compile again with `--explain errors` for more information
```

# Drawbacks

Changes in the error format can impact integration with other tools. For example, IDEs that use a
simple regex to detect the error would need to be updated to support the new format. This takes
time and community coordination.

While the new error format has a lot of benefits, it's possible that some errors will feel
"shoehorned" into it and, even after careful selection of secondary labels, may still not read as
well as the original format.

There is a fair amount of work involved to update the errors and explain text to the proposed
format.

# Alternatives

Rather than using the proposed error format format, we could only provide the verbose --explain
style that is proposed in this RFC. Respected programmers like
[John Carmack](https://twitter.com/ID_AA_Carmack/status/735197548034412546) have praised the Elm
error format.

```
Detected errors in 1 module.

-- TYPE MISMATCH ---------------------------------------------------------------
The right argument of (+) is causing a type mismatch.

25|       model + "1"
                  ^^^
(+) is expecting the right argument to be a:

    number

But the right argument is:

    String

Hint: To append strings in Elm, you need to use the (++) operator, not (+).
<http://package.elm-lang.org/packages/elm-lang/core/latest/Basics#++>

Hint: I always figure out the type of the left argument first and if it is acceptable on its own, I
assume it is "correct" in subsequent checks. So the problem may actually be in how the left and
right arguments interact.
```

*Example of an Elm error*

In developing this RFC, we experimented with both styles. The Elm error format is great as an
educational tool, and we wanted to leverage its style in Rust. For day-to-day work, though, we
favor an error format that puts heavy emphasis on quickly guiding the user to what the error is and
why it occurred, with an easy way to get the richer explanations (using --explain) when the user
wants them.

# Stabilization

Currently, this new rust error format is available on nightly using the
```export RUST_NEW_ERROR_FORMAT=true``` environment variable. Ultimately, this should become the
default. In order to get there, we need to ensure that the new error format is indeed an
improvement over the existing format in practice.

We also have not yet implemented the extended error format. This format will also be gated by its
own flag while we explore and stabilize it. Because of the relative difference in maturity here,
the default error message will be behind a flag for a cycle before it becomes default. The extended
error format will be implemented and a follow-up RFC will be posted describing its design. This will
start its stabilization period, after which time it too will be enabled.

How do we measure the readability of error messages?  This RFC details an educated guess as to what
would improve the current state but shows no ways to measure success.

Likewise, while some of us have been dogfooding these errors, we don't know what long-term use feels
like. For example, after a time does the use of color feel excessive?  We can always update the
errors as we go, but it'd be helpful to catch it early if possible.

# Unresolved questions

There are a few unresolved questions:
* Editors that rely on pattern-matching the compiler output will need to be updated. It's an open
question how best to transition to using the new errors. There is on-going discussion of
standardizing the JSON output, which could also be used.
* Can additional error notes be shown without the "rainbow problem" where too many colors and too
much boldness cause errors to become less readable?

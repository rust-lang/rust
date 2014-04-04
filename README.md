# Rust RFCs

Many changes, including bug fixes and documentation improvements can be 
implemented and reviewed via the normal GitHub pull request workflow.

Some changes though are "substantial", and we ask that these be put 
through a bit of a design process and produce a consensus among the Rust 
community and the [core team].

The "RFC" (request for comments process) is intended to provide a 
consistent and controlled path for new features to enter the language 
and standard libraries, so that all stakeholders can be confident about 
the direction the language is evolving in.

## When you need to follow this process

You need to follow this process if you intend to make "substantial" 
changes to the Rust distribution. What constitutes a "substantial" 
change is evolving based on community norms, but may include the following.

   - Any semantic or syntactic change to the language that is not a bugfix.
   - Changes to the interface between the compiler and libraries, 
including lang items and intrinsics.
   - Additions to `std`

Some changes do not require an RFC:

   - Rephrasing, reorganizing, refactoring, or otherwise "changing shape 
does not change meaning".
   - Additions that strictly improve objective, numerical quality 
criteria (warning removal, speedup, better platform coverage, more 
parallelism, trap more errors, etc.)
   - Additions only likely to be _noticed by_ other developers-of-rust, 
invisible to users-of-rust.

If you submit a pull request to implement a new feature without going 
through the RFC process, it may be closed with a polite request to 
submit an RFC first.

## What the process is

In short, to get a major feature added to Rust, one must first get the 
RFC merged into the RFC repo as a markdown file. At that point the RFC 
is 'active' and may be implemented with the goal of eventual inclusion 
into Rust.

* Fork the RFC repo http://github.com/rust-lang/rfcs
* Copy `0000-template.md` to `active/0000-my-feature.md` (where 
'my-feature' is descriptive. don't assign an RFC number yet).
* Fill in the RFC
* Submit a pull request. The pull request is the time to get review of 
the design from the larger community.
* Build consensus and integrate feedback. RFCs that have broad support 
are much more likely to make progress than those that don't receive any 
comments.
* Eventually, somebody on the [core team] will either accept the RFC by 
merging the pull request and assigning the RFC a number, at which point 
the RFC is 'active', or reject it by closing the pull request.

Once an RFC becomes active then authors may implement it and submit the 
feature as a pull request to the Rust repo. An 'active' is not a rubber 
stamp, and in particular still does not mean the feature will ultimately 
be merged; it does mean that in principle all the major stakeholders 
have agreed to the feature and are amenable to merging it.

Modifications to active RFC's can be done in followup PR's. An RFC that 
makes it through the entire process to implementation is considered 
'complete' and is moved to the 'complete' folder; an RFC that fails 
after becoming active is 'inactive' and moves to the 'inactive' folder.

### Help this is all too informal!

The process is intended to be as lightweight as reasonable for the 
present circumstances. As usual, we are trying to let the process be 
driven by consensus and community norms, not impose more structure than 
necessary.

[core team]: https://github.com/mozilla/rust/wiki/Note-core-team

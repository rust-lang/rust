# Rust Analyzer Roadmap 01

Written on 2018-11-06, extends approximately to February 2019.
After that, we should coordinate with the compiler/rls developers to align goals and share code and experience.


# Overall Goals

The mission is:
  * Provide an excellent "code analyzed as you type" IDE experience for the Rust language,
  * Implement the bulk of the features in Rust itself.


High-level architecture constraints:
  * Long-term, replace the current rustc frontend.
    It's *obvious* that the code should be shared, but OTOH, all great IDEs started as from-scratch rewrites.
  * Don't hard-code a particular protocol or mode of operation.
    Produce a library which could be used for implementing an LSP server, or for in-process embedding.
  * As long as possible, stick with stable Rust (NB: we currently use beta for 2018 edition and salsa).


# Current Goals

Ideally, we would be coordinating with the compiler/rls teams, but they are busy working on making Rust 2018 at the moment.
The sync-up point will happen some time after the edition, probably early 2019.
In the meantime, the goal is to **experiment**, specifically, to figure out how a from-scratch written RLS might look like.


## Data Storage and Protocol implementation

The fundamental part of any architecture is who owns which data, how the data is mutated and how the data is exposed to user.
For storage we use the [salsa](http://github.com/salsa-rs/salsa) library, which provides a solid model that seems to be the way to go.

Modification to source files is mostly driven by the language client, but we also should support watching the file system. The current
file watching implementation is a stub.

**Action Item:** implement reliable file watching service.

We also should extract LSP bits as a reusable library. There's already `gen_lsp_server`, but it is pretty limited.

**Action Item:** try using `gen_lsp_server` in more than one language server, for example for TOML and Nix.

The ideal architecture for `gen_lsp_server` is still unclear. I'd rather avoid futures: they bring significant runtime complexity
(call stacks become insane) and the performance benefits are negligible for our use case (one thread per request is perfectly OK given
the low amount of requests a language server receives). The current interface is based on crossbeam-channel, but it's not clear
if that is the best choice.


## Low-effort, high payoff features

Implementing 20% of type inference will give use 80% of completion.
Thus it makes sense to partially implement name resolution, type inference and trait matching, even though there is a chance that
this code is replaced later on when we integrate with the compiler

Specifically, we need to:

* **Action Item:** implement path resolution, so that we get completion in imports and such.
* **Action Item:** implement simple type inference, so that we get completion for inherent methods.
* **Action Item:** implement nicer completion infrastructure, so that we have icons, snippets, doc comments, after insert callbacks, ...


## Dragons to kill

To make experiments most effective, we should try to prototype solutions for the hardest problems.
In the case of Rust, the two hardest problems are:
  * Conditional compilation and source/model mismatch.
    A single source file might correspond to several entities in the semantic model.
    For example, different cfg flags produce effectively different crates from the same source.
  * Macros are intertwined with name resolution in a single fix-point iteration algorithm.
    This is just plain hard to implement, but also interacts poorly with on-demand.


For the first bullet point, we need to design descriptors infra and explicit mapping step between sources and semantic model, which is intentionally fuzzy in one direction.
The **action item** here is basically "write code, see what works, keep high-level picture in mind".

For the second bullet point, there's hope that salsa with its deep memoization will result in a fast enough solution even without being fully on-demand.
Again, the **action item** is to write the code and see what works. Salsa itself uses macros heavily, so it should be a great test.

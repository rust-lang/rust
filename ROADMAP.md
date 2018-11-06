# Rust Analyzer Roadmap 01

Written on 2018-11-06, extends approximately to February 2019.
After, we should coordinate with rustc/RLS developers to align goals and share code and experience.


# Overall Goals

The mission is:
  * provide an excellent "code analyzed as you type" IDE experience for the Rust language,
  * implementing the bulk of the features in Rust itself.


High-level architecture constraints:
  * Aim, long-term, to use the single code base with rustc.
    It's *obvious* then the code should be shared, but OTOH, all great IDEs started as from-scratch rewrites.
  * Don't hard-code a particular protocol or mode of operation.
    Produce a library which could be used for implementing an LSP server, or for in-process embedding.
  * As long as possible, stick with stable Rust (NB: we currently use beta for 2018 edition and salsa).


# Current Goals

We really should be coordinating with compiler/rls teams, but they are busy working on making Rust 2018 at the moment.
The sync-up point will happen some time after the edition, probably early next year.
In the mean time, the goal is to **experiment**, specifically, to figure out how a from-scratch written RLS might look like.


## Data Storage and Protocol implementation

The fundamental part of any architecture is who owned what data, how the data is mutated and how the data is exposed to user.
For storage we use the [salsa](http://github.com/salsa-rs/salsa) library, and the model already seems solid.

Most of modification is driven by the language client, but we also should support watching the file system, current implementation is a stub.

**Action Item:** implement reliable file watching service.

We also should extract LSP bits as a reusable library. There's already `gen_lsp_server`, but it is pretty limited.

**Action Item:** try using `gen_lsp_server` in more than one language server, for example for TOML and Nix.

Note that it is unclear what shape is ideal for `gen_lsp_server`.
I'd rather avoid futures: they bring significant runtime complexity (call stacks become insane), but we don't have c10k problem to solve.
Current interface is based on crossbeam-channel, but it's not clear if that is the best choice.


## Low-effort, high payoff features

Implementing 20% of type inference will give use 80% of completion.
Thus it makes sense to try to get some name resolution, type inference and trait matching implemented, even if all this code will be removed when we start sharing code with compiler.

Specifically, we need to:

**Action Item:** implement path resolution, so that we get completion in imports and such.
**Action Item:** implement simple type inference, so that we get completion for inherent methods.
**Action Item:** implement nicer completion infrastructure, so that we have icons, snippets, doc comments, after insert callbacks, ...


## Dragons to kill

To make experiments most effective, we should try to prototype solutions for the hardest problems.
In case of Rust, the two hardest problems are:
  * Conditional compilation and source/model mismatch.
    A single source file might correspond to several entities in the semantic model.
    For example, different cfg flags produce effectively different crates from the same source.
  * Macros are intertwined with name resolution in a single fix-point iteration algorithm.
    This is just plain hard to implement, but also interacts poorly with on-demand.


For the first bullet point, we need to design descriptors infra and explicit mapping step between sources and semantic model, which is intentionally fuzzy in one direction.
The action item here is basically "write code, see what works, keep high-level picture in mind".

For the second bullet point there's hope that salsa with its deep memorization will give us fast enough solution even without full on-demand.
Again, action item is to write the code and see what works. Salsa itself uses macros heavily, so it should be a great test.

- Feature Name: A plan for deprecating APIs within Rust
- Start Date: 2015-06-03
- RFC PR: 
- Rust Issue: 

# Summary

There has been an ongoing [discussion on internals](https://internals.rust-lang.org/t/thoughts-on-aggressive-deprecation-in-libstd/2176/55) about how we are going to evolve the standard library. This RFC tries to condense the consensus.

# Motivation

We want to guide the deprecation efforts to allow std to evolve freely to get the best possible API while ensuring minimum-breakage backwards compatibility for users and allow std authors to remove API items for a given version of Rust. Basically have our cake and eat it. Yum, cake.

Of course we cannot really keep and remove a feature at the same time. 
To square this circle, we can follow the process outlined herein.

# Detailed design

We already declare deprecation in terms of Rust versions (like "1.0", 
"1.2"). The current attribute looks like `#[deprecated(since = "1.0.0", 
reason="foo")]`. This should be extended to add an optional 
`removed_at` key, to state that the item should be made inaccessible at 
that version. Note that while this allows for marking items as 
deprecated, there is purposely no provision to actually *remove* items. 
In fact this proposal bans removing an API type outright, unless 
security concerns are deemed more important than the resulting breakage 
from removing it or the API item has some fault that means it cannot be 
used correctly at all (thus leaving the API in place would result in 
the same level of breakage than removing it).

Currently every rustc version implements only its own version, having 
multiple versions is possible using something like multirust, though 
this does not work within a build. Also currently rustc versions do not 
guarantee interoperability. This RFC aims to change this situation.

First, crates should state their target version using a `#![version = 
"1.0.0"]` attribute. Cargo should insert the current rust version by 
default on `cargo new` and *warn* if no version is defined on all other 
commands. It may optionally *note* that the specified target version is 
outdated on `cargo package`. To get the current rust version, cargo
could query rustc -V (with some postprocessing) or use some as yet
undefined symbol exported by the rust libraries.

[crates.io](https://crates.io) may deny 
packages that do not declare a version to give the target version 
requirement more weight to library authors. Cargo should also be able 
to hold back a new library version if its declared target version is 
newer than the rust version installed on the system. In those cases, 
cargo should emit a warning urging the user to upgrade their rust 
installation.

`rustc` should use this target version definition to check for 
deprecated items. If no target version is defined, deprecation checking 
is deactivated (as we cannot assume a specific rust version), however a 
warning stating the same should be issued (as with cargo – we should 
probably make cargo not warn on build to get rid of duplicate 
warnings). Otherwise, use of API items whose `since` attribute is less 
or equal to the target version of the crate should trigger a warning, 
while API items whose `removed_at` attribute is less or equal to the 
target version should trigger an error. 

Also if the target definition has a higher version than `rustc`, it
should warn that it probably has to be updated in order to build the
crate.

`rustdoc` should mark deprecated APIs as such (e.g. make them in a 
lighter gray font) and relegate removed APIs to a section below all 
others (and that may be hidden via a checkbox). We should not 
completely remove the documentation, as users of libraries that target 
old versions may still have a use for them, but neither should we let 
them clutter the docs. 

# Drawbacks

By requiring full backwards-compatibility, we will never be able to 
actually remove stuff from the APIs, which will probably lead to some 
bloat. Other successful languages have lived with this for multiple 
decades, so it appears the tradeoff has seen some confirmation already. 

# Alternatives

* Follow a more agressive strategy that actually removes stuff from the 
API. This would make it easier for the libstd creators at some cost for 
library and application writers, as they are required to keep up to 
date or face breakage * Hide deprecated items in the docs: This could 
be done either by putting them into a linked extra page or by adding a 
"show deprecated" checkbox that may be default be checked or not, 
depending on who you ask. This will however confuse people, who see the 
deprecated APIs in some code, but cannot find them in the docs anymore 
* Allow to distinguish "soft" and "hard" deprecation, so that an API 
can be marked as "soft" deprecated to dissuade new uses before hard 
deprecation is decided. Allowing people to specify deprecation in 
future version appears to have much of the same benefits without 
needing a new attribute key. * Decide deprecation on a per-case basis. 
This is what we do now. The proposal just adds a well-defined process 
to it * Never deprecate anything. Evolve the API by adding stuff only. 
Rust would be crushed by the weight of its own cruft before 2.0 even 
has a chance to land. Users will be uncertain which APIs to use * We 
could extend the deprecation feature to cover libraries. As Cargo.toml 
already defines the target versions of dependencies (unless declared as 
`"*"`), we could use much of the same machinery to allow library 
authors to join the process

# Unresolved questions

Should we allow library writers to use the same features for 
deprecating their API items?

- Start Date: 2014-12-18
- RFC PR: [531](https://github.com/rust-lang/rfcs/pull/531)
- Rust Issue: n/a

# Summary

According to current documents, the RFC process is required to make "substantial" changes to the Rust 
distribution. It is currently lightweight, but lacks a definition for the Rust distribution. This RFC 
aims to amend the process with a both broad and clear definition of "Rust distribution," while still 
keeping the process itself in tact.

# Motivation

The motivation for this change comes from the recent decision for Crates.io to affirm its first come,
first serve policy. While there was discussion of the matter on a GitHub issue, this discussion was
rather low visibility. Regardless of the outcome of this particular decision, it highlights the 
fact that there is not a clear place for thorough discussion of policy decisions related to the 
outermost parts of Rust.

# Detailed design

To remedy this issue, there must be a defined scope for the RFC process. This definition would be 
incorporated into the section titled "When you need to follow this process." The goal here is to be as
explicit as possible. This RFC proposes that the scope of the RFC process be defined as follows:

* Rust
* Cargo
* Crates.io
* The RFC process itself

This definition explicitly does not include:

* Other crates maintained under the rust-lang organization, such as time.

# Drawbacks

The only particular drawback would be if this definition is too narrow, it might be restrictive.
However, this definition fortunately includes the ability to amend the RFC process. So, this
could be expanded if the need exists.

# Alternatives

The alternative is leaving the process as is. However, adding clarity at little to no cost should
be preferred as it lowers the barrier to entry for contributions, and increases the visibility of
potential changes that may have previously been discussed outside of an RFC.

# Unresolved questions

Are there other things that should be explicitly included as part of the scope of the RFC process right now?

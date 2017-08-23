- Feature Name: N/A
- Start Date: 2017-04-26
- RFC PR: https://github.com/rust-lang/rfcs/pull/1983
- Rust Issue: N/A

# Summary
[summary]: #summary

Amend [RFC 1242] to require an RFC for deprecation of crates from the 
rust-lang-nursery.

[RFC 1242]: https://github.com/rust-lang/rfcs/blob/master/text/1242-rust-lang-crates.md

# Motivation
[motivation]: #motivation

There are currently very ubiquitous crates in the nursery that are being used 
by lots and lots of people, as evidenced by the crates.io download numbers (for 
lack of a better popularity metric):

| Nursery crate | Downloads |
| ------------- | --------- |
| bitflags      |    3,156k |
| rand          |    2,615k |
| log           |    2,417k |
| lazy-static   |    2,108k |
| tempdir       |      934k |
| uuid          |      759k |
| glob          |      467k |
| net2          |      452k |
| getopts       |      452k |
| rustfmt       |       80k |
| simd          |       14k |

(numbers as of 2017-04-26)

[RFC 1242] currently specifies that

> The libs subteam can deprecate nursery crates at any time

The libs team can of course be trusted to be judicious in making such 
decisions. However, considering that many of the nursery crates are depended on 
by big fractions of the Rust ecosystem, suddenly deprecating things without 
public discussion seems contrary to Rust's philosophy of stability and 
community participation. Involving the Rust community at large in these 
decisions offers the benefits of the RFC process such as increased visibility, 
differing viewpoints, and transparency.

# Detailed design
[design]: #detailed-design

The exact amendment is included as a change to the RFC in this PR.
[View the amended text](1242-rust-lang-crates.md).

# How We Teach This
[how-we-teach-this]: #how-we-teach-this

N/A

# Drawbacks
[drawbacks]: #drawbacks

Requiring an RFC for deprecation might impose an undue burden on the library 
subteam in terms of crate maintenance. However, as [RFC 1242] states, this is
not a major commitment.

Acceptance into the nursery could be hindered if it is believed it could be 
hard to reverse course later due to the required RFC being percieved as an 
obstacle. On the other hand, RFCs with broad consensus do not generally impose 
a large procedural burden, and if there is no consensus it might be too early 
to deprecate a nursery crate anyway.

# Alternatives
[alternatives]: #alternatives

Don't change the process and let the library subteam make deprecation decisions 
for nursery crates.

# Unresolved questions
[unresolved]: #unresolved-questions

None as of yet.

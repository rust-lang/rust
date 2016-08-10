- Feature Name: N/A
- Start Date: 2016-07-21
- RFC PR: https://github.com/rust-lang/rfcs/pull/1683
- Rust Issue: N/A

# Summary
[summary]: #summary

Create a team responsible for documentation for the Rust project.

# Motivation
[motivation]: #motivation

[RFC 1068] introduced a federated governance model for the Rust project. Several initial subteams were set up. There was a note
after the [original subteam list] saying this:

[RFC 1068]: https://github.com/rust-lang/rfcs/blob/master/text/1068-rust-governance.md
[original subteam list]: https://github.com/rust-lang/rfcs/blob/master/text/1068-rust-governance.md#the-teams

> In the long run, we will likely also want teams for documentation and for community events, but these can be spun up once there is a more clear need (and available resources).

Now is the time for a documentation subteam.

## Why documentation was left out

Documentation was left out of the original list because it wasn't clear that there would be anyone but me on it. Furthermore,
one of the original reasons for the subteams was to decide who gets counted amongst consensus for RFCs, but it was unclear
how many documentation-related RFCs there would even be.

## Chicken, meet egg

However, RFCs are not only what subteams do. To quote the RFC:

> * Shepherding RFCs for the subteam area. As always, that means (1) ensuring
>   that stakeholders are aware of the RFC, (2) working to tease out various
>   design tradeoffs and alternatives, and (3) helping build consensus.
> * Accepting or rejecting RFCs in the subteam area.
> * Setting policy on what changes in the subteam area require RFCs, and reviewing direct PRs for changes that do not require an RFC.
> * Delegating reviewer rights for the subteam area. The ability to r+ is not limited to team members, and in fact earning r+ rights is a good stepping stone toward team membership. Each team should set reviewing policy, manage reviewing rights, and ensure that reviews take place in a timely manner. (Thanks to Nick Cameron for this suggestion.)

The first two are about RFCs themselves, but the second two are more pertinent to documentation. In particular,
deciding who gets `r+` rights is important. A lack of clarity in this area has been unfortuante, and has led to a
chicken and egg situation: without a documentation team, it's unclear how to be more involved in working on Rust's
documentation, but without people to be on the team, there's no reason to form a team. For this reason, I think
a small initial team will break this logjam, and provide room for new contributors to grow.

# Detailed design
[design]: #detailed-design

The Rust documentation team will be responsible for all of the things listed above. Specifically, they will pertain
to these areas of the Rust project:

* The standard library documentation
* The book and other long-form docs
* Cargo's documentation
* The Error Index

Furthermore, the documentation team will be available to help with ecosystem documentation, in a few ways. Firstly,
in an advisory capacity: helping people who want better documentation for their crates to understand how to accomplish
that goal. Furthermore, monitoring the overall ecosystem documentation, and identifying places where we could contribute
and make a large impact for all Rustaceans. If the Rust project itself has wonderful docs, but the ecosystem has terrible
docs, then people will still be frustrated with Rust's documentation situation, especially given our anti-batteries-included
attitude. To be clear, this does not mean _owning_ the ecosystem docs, but rather working to contribute in more ways
than just the Rust project itself.

We will coordinate in the `#rust-docs` IRC room, and have regular meetings, as the team sees fit. Regular meetings will be
important to coordinate broader goals; and participation will be important for team members. We hold meetings weekly.

## Membership

* @steveklabnik, team lead
* @GuillaumeGomez
* @jonathandturner
* @peschkaj

It's important to have a path towards attaining team membership; there are some other people who have already been doing
docs work that aren't on this list. These guidelines are not hard and fast, however, anyone wanting to eventually be a
member of the team should pursue these goals:

* Contributing documentation patches to Rust itself
* Attending doc team meetings, which are open to all
* generally being available on [IRC][^IRC] to collaborate with others

I am not quantifying this exactly because it's not about reaching some specific number; adding someone to the team should
make sense if someone is doing all of these things.

[^IRC]: The #rust-docs channel on irc.mozilla.org

# Drawbacks
[drawbacks]: #drawbacks

This is Yet Another Team. Do we have too many teams? I don't think so, but someone might.

# Alternatives
[alternatives]: #alternatives

The main alternative is not having a team. This is the status quo, so the situation is well-understood.

It's possible that docs come under the purvew of "tools", and so maybe the docs team would be an expansion
of the tools team, rather than its own new team. Or some other subteam.

# Unresolved questions
[unresolved]: #unresolved-questions

None.

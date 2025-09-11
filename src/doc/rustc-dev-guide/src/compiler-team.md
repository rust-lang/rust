# About the compiler team

rustc is maintained by the [Rust compiler team][team]. The people who belong to
this team collectively work to track regressions and implement new features.
Members of the Rust compiler team are people who have made significant
contributions to rustc and its design.

[team]: https://www.rust-lang.org/governance/teams/compiler

## Discussion

Currently the compiler team chats in Zulip:

- Team chat occurs in the [`t-compiler`][zulip-t-compiler] stream on the Zulip instance
- There are also a number of other associated Zulip channels,
  such as [`t-compiler/help`][zulip-help], where people can ask for help
  with rustc development, or [`t-compiler/meetings`][zulip-meetings],
  where the team holds their weekly triage and steering meetings.

## Reviewers

If you're interested in figuring out who can answer questions about a
particular part of the compiler, or you'd just like to know who works on what,
check out [triagebot.toml's assign section][map].
It contains a listing of the various parts of the compiler and a list of people
who are reviewers of each part.

[map]: https://github.com/rust-lang/rust/blob/master/triagebot.toml

## Rust compiler meeting

The compiler team has a weekly meeting where we do triage and try to
generally stay on top of new bugs, regressions, and discuss important
things in general.
They are held on [Zulip][zulip-meetings]. It works roughly as follows:

- **Announcements, MCPs/FCPs, and WG-check-ins:** We share some
  announcements with the rest of the team about important things we want
  everyone to be aware of. We also share the status of MCPs and FCPs and we
  use the opportunity to have a couple of WGs giving us an update about
  their work.
- **Check for beta and stable nominations:** These are nominations of things to
  backport to beta and stable respectively.
  We then look for new cases where the compiler broke previously working
  code in the wild. Regressions are important issues to fix, so it's
  likely that they are tagged as P-critical or P-high; the major
  exception would be bug fixes (though even there we often [aim to give
  warnings first][procedure]).
- **Review P-critical and P-high bugs:** P-critical and P-high bugs are
  those that are sufficiently important for us to actively track
  progress. P-critical and P-high bugs should ideally always have an
  assignee.
- **Check S-waiting-on-team and I-nominated issues:** These are issues where feedback from
  the team is desired.
- **Look over the performance triage report:** We check for PRs that made the
    performance worse and try to decide if it's worth reverting the performance regression or if
    the regression can be addressed in a future PR.

The meeting currently takes place on Thursdays at 10am Boston time
(UTC-4 typically, but daylight savings time sometimes makes things
complicated).

[procedure]: ./bug-fix-procedure.md
[zulip-t-compiler]: https://rust-lang.zulipchat.com/#narrow/stream/131828-t-compiler
[zulip-help]: https://rust-lang.zulipchat.com/#narrow/stream/182449-t-compiler.2Fhelp
[zulip-meetings]: https://rust-lang.zulipchat.com/#narrow/stream/238009-t-compiler.2Fmeetings

## Team membership

Membership in the Rust team is typically offered when someone has been
making significant contributions to the compiler for some
time. Membership is both a recognition but also an obligation:
compiler team members are generally expected to help with upkeep as
well as doing reviews and other work.

If you are interested in becoming a compiler team member, the first
thing to do is to start fixing some bugs, or get involved in a working
group. One good way to find bugs is to look for
[open issues tagged with E-easy](https://github.com/rust-lang/rust/issues?q=is%3Aopen+is%3Aissue+label%3AE-easy)
or
[E-mentor](https://github.com/rust-lang/rust/issues?q=is%3Aopen+is%3Aissue+label%3AE-mentor).

You can also dig through the graveyard of PRs that were
[closed due to inactivity](https://github.com/rust-lang/rust/pulls?q=is%3Apr+label%3AS-inactive),
some of them may contain work that is still useful - refer to the
associated issues, if any - and only needs some finishing touches
for which the original author didn't have time.

### r+ rights

Once you have made a number of individual PRs to rustc, we will often
offer r+ privileges. This means that you have the right to instruct
"bors" (the robot that manages which PRs get landed into rustc) to
merge a PR
([here are some instructions for how to talk to bors][homu-guide]).

[homu-guide]: https://bors.rust-lang.org/

The guidelines for reviewers are as follows:

- You are always welcome to review any PR, regardless of who it is
  assigned to.  However, do not r+ PRs unless:
  - You are confident in that part of the code.
  - You are confident that nobody else wants to review it first.
    - For example, sometimes people will express a desire to review a
      PR before it lands, perhaps because it touches a particularly
      sensitive part of the code.
- Always be polite when reviewing: you are a representative of the
  Rust project, so it is expected that you will go above and beyond
  when it comes to the [Code of Conduct].

[Code of Conduct]: https://www.rust-lang.org/policies/code-of-conduct

### Reviewer rotation

Once you have r+ rights, you can also be added to the [reviewer rotation].
[triagebot] is the bot that [automatically assigns] incoming PRs to reviewers.
If you are added, you will be randomly selected to review
PRs. If you find you are assigned a PR that you don't feel comfortable
reviewing, you can also leave a comment like `r? @so-and-so` to assign
to someone else — if you don't know who to request, just write `r?
@nikomatsakis for reassignment` and @nikomatsakis will pick someone
for you.

[reviewer rotation]: https://github.com/rust-lang/rust/blob/36285c5de8915ecc00d91ae0baa79a87ed5858d5/triagebot.toml#L528-L577
[triagebot]: https://github.com/rust-lang/triagebot/
[automatically assigns]: https://forge.rust-lang.org/triagebot/pr-assignment.html

Getting on the reviewer rotation is much appreciated as it lowers the
review burden for all of us! However, if you don't have time to give
people timely feedback on their PRs, it may be better that you don't
get on the list.

### Full team membership

Full team membership is typically extended once someone made many
contributions to the Rust compiler over time, ideally (but not
necessarily) to multiple areas. Sometimes this might be implementing a
new feature, but it is also important — perhaps more important! — to
have time and willingness to help out with general upkeep such as
bugfixes, tracking regressions, and other less glamorous work.

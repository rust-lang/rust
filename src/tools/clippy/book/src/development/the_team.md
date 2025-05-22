# The team

Everyone who contributes to Clippy makes the project what it is. Collaboration
and discussions are the lifeblood of every open-source project. Clippy has a
very flat hierarchy. The teams mainly have additional access rights to the repo.

This document outlines the onboarding process, as well as duties, and access
rights for members of a group.

All regular events mentioned in this chapter are tracked in the [calendar repository].
The calendar file is also available for download: [clippy.ics]

## Everyone

Everyone, including you, is welcome to join discussions and contribute in other
ways, like PRs.

You also have some triage rights, using `@rustbot` to add labels and claim
issues. See [labeling with @rustbot].

A rule for everyone should be to keep a healthy work-life balance. Take a break
when you need one.

## Clippy-Contributors

This is a group of regular contributors to Clippy to help with triaging.

### Duties

This team exists to make contributing easier for regular members. It doesn't
carry any duties that need to be done. However, we want to encourage members of
this group to help with triaging, which can include:

1. **Labeling issues**

    For the `good first issue` label, it can still be good to use `@rustbot` to
    subscribe to the issue and help interested parties, if they post questions
    in the comments. 

2. **Closing duplicate or resolved issues**

    When you manually close an issue, it's often a good idea, to add a short
    comment explaining the reason.

3. **Ping people after two weeks of inactivity**

    We try to keep issue assignments and PRs fairly up-to-date. After two weeks,
    it can be good to send a friendly ping to the delaying party.

    You might close a PR with the `I-inactive-closed` label if the author is
    busy or wants to abandon it. If the reviewer is busy, the PR can be
    reassigned to someone else.

    Checkout: https://triage.rust-lang.org/triage/rust-lang/rust-clippy to
    monitor PRs.

While not part of their duties, contributors are encouraged to review PRs
and help on Zulip. The team always appreciates help!

### Membership

If you have been contributing to Clippy for some time, we'll probably ask you if
you want to join this team. Members of this team are also welcome to suggest
people who they think would make a great addition to this group.

For this group, there is no direct onboarding process. You're welcome to just
continue what you've been doing. If you like, you can ask for someone to mentor
you, either in the Clippy stream on Zulip or privately via a PM.

If you have been inactive in Clippy for over three months, we'll probably move
you to the alumni group. You're always welcome to come back.

## The Clippy Team

[The Clippy team](https://www.rust-lang.org/governance/teams/dev-tools#team-clippy)
is responsible for maintaining Clippy.

### Duties

1. **Respond to PRs in a timely manner**

    It's totally fine, if you don't have the time for reviews right now.
    You can reassign the PR to a random member by commenting `r? clippy`.

2. **Take a break when you need one**

    You are valuable! Clippy wouldn't be what it is without you. So take a break
    early and recharge some energy when you need to.

3. **Be responsive on Zulip**

    This means in a reasonable time frame, so responding within one or two days
    is totally fine.

    It's also good, if you answer threads on Zulip and take part in our Clippy
    meetings, every two weeks. The meeting dates are tracked in the [calendar repository].
    

4. **Sync Clippy with the rust-lang/rust repo**

    This is done every two weeks, usually by @flip1995.

5. **Update the changelog**

    This needs to be done for every release, every six weeks.

### Membership

If you have been active for some time, we'll probably reach out and ask
if you want to help with reviews and eventually join the Clippy team.

During the onboarding process, you'll be assigned pull requests to review.
You'll also have an active team member as a mentor who'll stay in contact via
Zulip DMs to provide advice and feedback. If you have questions, you're always
welcome to ask, that is the best way to learn. Once you're done with the review,
you can ping your mentor for a full review and to r+ the PR in both of your names.

When your mentor is confident that you can handle reviews on your own, they'll
start an informal vote among the active team members to officially add you to
the team. This vote is usually accepted unanimously. Then you'll be added to
the team once you've confirmed that you're still interested in joining. The
onboarding phase typically takes a couple of weeks to a few months.

If you have been inactive in Clippy for over three months, we'll probably move
you to the alumni group. You're always welcome to come back.

[calendar repository]: https://github.com/rust-lang/calendar/blob/main/clippy.toml
[clippy.ics]: https://rust-lang.github.io/calendar/clippy.ics
[labeling with @rustbot]: https://forge.rust-lang.org/triagebot/labeling.html

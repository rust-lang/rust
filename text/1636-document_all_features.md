- Feature Name: document_all_features
- Start Date: 2016-06-03
- RFC PR: https://github.com/rust-lang/rfcs/pull/1636
- Rust Issue: https://github.com/rust-lang-nursery/reference/issues/9


# Summary

One of the major goals of Rust's development process is *stability without stagnation*. That means we add features regularly. However, it can be difficult to *use* those features if they are not publicly documented anywhere. Therefore, this RFC proposes requiring that all new language features and public standard library items must be documented before landing on the stable release branch (item documentation for the standard library; in the language reference for language features).


## Outline

-   Summary
    -   Outline
-   Motivation
    -   The Current Situation
    -   Precedent
-   Detailed design
    -   New RFC section: “How do we teach this?”
    -   New requirement to document changes before stabilizing
        -   Language features
            -   Reference
                -   The state of the reference
            -   _The Rust Programming Language_
        -   Standard library
-   How do we teach this?
-   Drawbacks
-   Alternatives
-   Unresolved questions


# Motivation

At present, new language features are often documented *only* in the RFCs which propose them and the associated announcement blog posts. Moreover, as features change, the existing official language documentation (the Rust Book, Rust by Example, and the language reference) can increasingly grow outdated.

Although the Rust Book and Rust by Example are kept relatively up to date, [the reference is not][home-to-reference]:

> While Rust does not have a specification, the reference tries to describe its working in detail. *It tends to be out of date.* (emphasis mine)

Importantly, though, this warning only appears on the [main site][home-to-reference], not in the reference itself. If someone searches for e.g. that `deprecated` attribute and *does* find the discussion of the deprecated attribute, they will have no reason to believe that the reference is wrong.

[home-to-reference]: https://www.rust-lang.org/documentation.html

For example, the change in Rust 1.9 to allow users to use the `#[deprecated]` attribute for their own libraries was, at the time of writing this RFC, *nowhere* reflected in official documentation. (Many other examples could be supplied; this one was chosen for its relative simplicity and recency.) The Book's [discussion of attributes][book-attributes] linked to the [reference list of attributes][ref-attributes], but as of the time of writing the reference [still specifies][ref-compiler-attributes] that `deprecated` was a compiler-only feature. The two places where users might have become aware of the change are [the Rust 1.9 release blog post][1.9-blog] and the [RFC itself][RFC-1270]. Neither (yet) ranked highly in search; users were likely to be misled.

[book-attributes]: https://doc.rust-lang.org/book/attributes.html
[ref-attributes]: https://doc.rust-lang.org/reference.html#attributes
[ref-compiler-attributes]: https://doc.rust-lang.org/reference.html#compiler-features
[1.9-blog]: http://blog.rust-lang.org/2016/05/26/Rust-1.9.html#deprecation-warnings
[RFC-1270]: https://github.com/rust-lang/rfcs/blob/master/text/1270-deprecation.md

Changing this to require all language features to be documented before stabilization would mean Rust users can use the language documentation with high confidence that it will provide exhaustive coverage of all stable Rust features.

Although the standard library is in excellent shape regarding documentation, including it in this policy will help guarantee that it remains so going forward.

## The Current Situation

Today, the canonical source of information about new language features is the RFCs which define them. The Rust Reference is substantially out of date, and not all new features have made their way into _The Rust Programming Language_.

There are several serious problems with the _status quo_ of using RFCs as ad hoc documentation:

1. Many users of Rust may simply not know that these RFCs exist. The number of users who do not know (or especially care) about the RFC process or its history will only increase as Rust becomes more popular.

2. In many cases, especially in more complicated language features, some important elements of the decision, details of implementation, and expected behavior are fleshed out either in the pull-request discussion for the RFC, or in the implementation issues which follow them.

3. The RFCs themselves, and even more so the associated pull request discussions, are often dense with programming language theory. This is as it should be in context, but it means that the relevant information may be inaccessible to Rust users without prior PLT background, or without the patience to wade through it.

4. Similarly, information about the final decisions on language features is often buried deep at the end of long and winding threads (especially for a complicated feature like `impl` specialization).

5. Information on how the features will be used is often closely coupled to information on how the features will be implemented, both in the RFCs and in the discussion threads. Again, this is as it should be, but it makes it difficult (at best!) for ordinary Rust users to read.

In short, RFCs are a poor source of information about language features for the ordinary Rust user. Rust users should not need to be troubled with details of how the language is implemented works simply to learn how pieces of it work. Nor should they need to dig through tens (much less hundreds) of comments to determine what the final form of the feature is.

However, there is currently no other documentation at all for many newer features. This is a significant barrier to adoption of the language, and equally of adoption of new features which will improve the ergonomics of the language.

## Precedent

This exact idea has been adopted by the Ember community after their somewhat bumpy transitions at the end of their 1.x cycle and leading into their 2.x transition. As one commenter there [put it][@davidgoli]:

> The fact that 1.13 was released without updated guides is really discouraging to me as an Ember adopter. It may be much faster, the features may be much cooler, but to me, they don't exist unless I can learn how to use them from documentation. Documentation IS feature work. ([@davidgoli])

[@davidgoli]: https://github.com/emberjs/rfcs/pull/56#issuecomment-114635962

The Ember core team agreed, and embraced the principle outlined in [this comment][@guarav0]:

> No version shall be released until guides and versioned API documentation is ready. This will allow newcomers the ability to understand the latest release. ([@guarav0])

[@guarav0]: https://github.com/emberjs/rfcs/pull/56#issuecomment-114339423

One of the main reasons not to adopt this approach, that it might block features from landing as soon as they otherwise might, was [addressed][@eccegordo] in that discussion as well:

> Now if this documentation effort holds up the releases people are going to grumble. But so be it. The challenge will be to effectively parcel out the effort and relieve the core team to do what they do best. No single person should be a gate. But lack of good documentation should gate releases. That way a lot of eyes are forced to focus on the problem. We can't get the great new toys unless everybody can enjoy the toys. ([@eccegordo])

[@eccegordo]: https://github.com/emberjs/rfcs/pull/56#issuecomment-114389963

The basic decision has led to a substantial improvement in the currency of the documentation (which is now updated the same day as a new version is released). Moreover, it has spurred ongoing development of better tooling around documentation to manage these releases. Finally, at least in the RFC author's estimation, it has also led to a substantial increase in the overall quality of that documentation, possibly as a consequence of increasing the community involvement in the documentation process (including the formation of a documentation subteam).


# Detailed design

The basic process of developing new language features will remain largely the same as today. The required changes are two additions:

- a new section in the RFC, "How do we teach this?" modeled on Ember's updated RFC process

- a new requirement that the changes themselves be properly documented before being merged to stable


## New RFC section: "How do we teach this?"

Following the example of Ember.js, we must add a new section to the RFC, just after **Detailed design**, titled **How do we teach this?** The section should explain what changes need to be made to documentation, and if the feature substantially changes what would be considered the "best" way to solve a problem or is a fairly mainstream issue, discuss how it might be incorporated into _The Rust Programming Language_ and/or _Rust by Example_.

Here is the Ember RFC section, with appropriate substitutions and modifications:

> # How We Teach This
> What names and terminology work best for these concepts and why? How is this idea best presented? As a continuation of existing Rust patterns, or as a wholly new one?
>
> Would the acceptance of this proposal change how Rust is taught to new users at any level?  What additions or changes to the Rust Reference, _The Rust Programing Language_, and/or _Rust by Example_ does it entail?
>
> How should this feature be introduced and taught to existing Rust users?

For a great example of this in practice, see the (currently open) [Ember RFC: Module Unification], which includes several sections discussing conventions, tooling, concepts, and impacts on testing.

[Ember RFC: Module Unification]: https://github.com/dgeb/rfcs/blob/module-unification/text/0000-module-unification.md#how-we-teach-this

## New requirement to document changes before stabilizing

[require-documentation-before-stabilization]: #new-requirement-to-document-changes-before-stabilizing

Prior to stabilizing a feature, the features will now be documented as follows:

- Language features:
    - must be documented in the Rust Reference.
    - should be documented in _The Rust Programming Language_.
    - may be documented in _Rust by Example_.
- Standard library additions must include documentation in `std` API docs.
- Both language features and standard library changes must include:
    - a single line for the changelog
    - a longer summary for the long-form release announcement.

Stabilization of a feature must not proceed until the requirements outlined in the **How We Teach This** section of the originating RFC have been fulfilled.

### Language features

We will document *all* language features in the Rust Reference, as well as updating _The Rust Programming Language_ and _Rust by Example_ as appropriate. (Not all features or changes will require updates to the books.)

#### Reference

[reference]: #reference

This will necessarily be a manual process, involving updates to the `reference.md` file. (It may at some point be sensible to break up the Reference file for easier maintenance; that is left aside as orthogonal to this discussion.)

Feature documentation does not need to be written by the feature author. In fact, this is one of the areas where the community may be most able to support the language/compiler developers even if not themselves programming language theorists or compiler hackers. This may free up the compiler developers' time. It will also help communicate the features in a way that is accessible to ordinary Rust users.

New features do not need to be documented to be merged into `master`/nightly

Instead, the documentation process should immediately precede the move to stabilize. Once the *feature* has been deemed ready for stabilization, either the author or a community volunteer should write the *reference material* for the feature, to be incorporated into the Rust Reference.

The reference material need not be especially long, but it should be long enough for ordinary users to learn how to use the language feature *without reading the RFCs*.

Discussion of stabilizing a feature in a given release will now include the status of the reference material.

##### The current state of the reference

[refstate]: #the-current-state-of-the-reference

Since the reference is fairly out of date, we should create a "strike team" to update it. This can proceed in parallel with the documentation of new features.

Updating the reference should proceed stepwise:

1. Begin by adding an appendix in the reference with links to all accepted RFCs which have been implemented but are not yet referenced in the documentation.
2. As the reference material is written for each of those RFC features, remove it from that appendix.

The current presentation of the reference is also in need of improvement: a single web page with *all* of this content is difficult to navigate, or to update. Therefore, the strike team may also take this opportunity to reorganize the reference and update its presentation.

#### _The Rust Programming Language_

[trpl]: #the-rust-programming-language

Most new language features should be added to _The Rust Programming Language_. However, since the book is planned to go to print, the main text of the book is expected to be fixed between major revisions. As such, new features should be documented in an online appendix to the book, which may be titled e.g. "Newest Features."

The published version of the book should note that changes and languages features made available after the book went to print will be documented in that online appendix.

### Standard library

In the case of the standard library, this could conceivably be managed by setting the `#[forbid(missing_docs)]` attribute on the library roots. In lieu of that, manual code review and general discipline should continue to serve. However, if automated tools *can* be employed here, they should.

# How do we teach this?

Since this RFC promotes including this section, it includes it itself. (RFCs, unlike Rust `struct` or `enum` types, may be freely self-referential. No boxing required.)

To be most effective, this will involve some changes both at a process and core-team level, and at a community level.

1. The RFC template must be updated to include the new section for teaching.
2. The RFC process in the [RFCs README] must be updated, specifically by including "fail to include a plan for documenting the feature" in the list of possible problems in "Submit a pull request step" in [What the process is].
3. Make documentation and teachability of new features *equally* high priority with the features themselves, and communicate this clearly in discussion of the features. (Much of the community is already very good about including this in considerations of language design; this simply makes this an explicit goal of discussions around RFCs.)

[RFCs README]: https://github.com/rust-lang/rfcs/blob/master/README.md
[What the process is]: https://github.com/rust-lang/rfcs/blob/master/README.md#what-the-process-is

This is also an opportunity to allow/enable community members with less experience to contribute more actively to _The Rust Programming Language_, _Rust by Example_, and the Rust Reference.

1. We should write issues for feature documentation, and may flag them as approachable entry points for new users.

2. We may use the more complicated language reference issues as points for mentoring developers interested in contributing to the compiler. Helping document a complex language feature may be a useful on-ramp for working on the compiler itself.

At a "messaging" level, we should continue to emphasize that *documentation is just as valuable as code*. For example (and there are many other similar opportunities): in addition to highlighting new language features in the release notes for each version, we might highlight any part of the documentation which saw substantial improvement in the release.


# Drawbacks

1.  The largest drawback at present is that the language reference is *already* quite out of date. It may take substantial work to get it up to date so that new changes can be landed appropriately. (Arguably, however, this should be done regardless, since the language reference is an important part of the language ecosystem.)

2.  Another potential issue is that some sections of the reference are particularly thorny and must be handled with considerable care (e.g. lifetimes). Although in general it would not be necessary for the author of the new language feature to write all the documentation, considerable extra care and oversight would need to be in place for these sections.

3.  This may delay landing features on stable. However, all the points raised in **Precedent** on this apply, especially:

    > We can't get the great new toys unless everybody can enjoy the toys. ([@eccegordo])

    For Rust to attain its goal of *stability without stagnation*, its documentation must also be stable and not stagnant.

4.  If the forthcoming docs team is unable to provide significant support, and perhaps equally if the rest of the community does not also increase involvement, this will simply not work. No individual can manage all of these docs alone.


# Alternatives

-   **Just add the "How do we teach this?" section.**

    Of all the alternatives, this is the easiest (and probably the best). It does not substantially change the state with regard to the documentation, and even having the section in the RFC does not mean that it will end up added to the docs, as evidence by the [`#[deprecated]` RFC][RFC 1270], which included as part of its text:

    > The language reference will be extended to describe this feature as outlined in this RFC. Authors shall be advised to leave their users enough time to react before removing a deprecated item.

    This is not a small downside by any stretch—but adding the section to the RFC will still have all the secondary benefits noted above, and it probably at least somewhat increases the likelihood that new features do get documented.

-   **Embrace the documentation, but do not include "How do we teach this?" section in new RFCs.**

      This still gives us most of the benefits (and was in fact the original form of the proposal), and does not place a new burden on RFC authors to make sure that knowing how to *teach* something is part of any new language or standard library feature.

      On the other hand, thinking about the impact on teaching should further improve consideration of the general ergonomics of a proposed feature. If something cannot be *taught* well, it's likely the design needs further refinement.

-   **No change; leave RFCs as canonical documentation.**

      This approach can take (at least) two forms:


    1. We can leave things as they are, where the RFC and surrounding discussion form the primary point of documentation for newer-than-1.0 language features. As part of that, we could just link more prominently to the RFC repository and describe the process from the documentation pages.
    2. We could automatically render the text of the RFCs into part of the documentation used on the site (via submodules and the existing tooling around Markdown documents used for Rust documentation).

    However, for all the reasons highlighted above in **Motivation: The Current Situation**, RFCs and their associated threads are *not* a good canonical source of information on language features.

-   **Add a rule for the standard library but not for language features.**

    This would basically just turn the _status quo_ into an official policy. It has all the same drawbacks as no change at all, but with the possible benefit of enabling automated checks on standard library documentation.

-   **Add a rule for language features but not for the standard library.**

      The standard library is in much better shape, in no small part because of the ease of writing inline documentation for new modules. Adding a formal rule may not be necessary if good habits are already in place.

      On the other hand, having a formal policy would not seem to *hurt* anything here; it would simply formalize what is already happening (and perhaps, via linting attributes, make it easy to spot when it has failed).

-   **Eliminate the reference entirely.**

      Since the reference is already substantially out of date, it might make sense to stop presenting it publicly at all, at least until such a time as it has been completely reworked and updated.

      The main upside to this is the reality that an outdated and inaccurate reference may be worse than no reference at all, as it may mislead espiecally new Rust users.

      The main downside, of course, is that this would leave very large swaths of the language basically without *any* documentation, and even more of it only documented in RFCs than is the case today.


[RFC 1270]: https://github.com/rust-lang/rfcs/pull/1270

# Unresolved questions

- How do we clearly distinguish between features on nightly, beta, and stable Rust—in the reference especially, but also in the book?
- For the standard library, once it migrates to a crates structure, should it simply include the `#[forbid(missing_docs)]` attribute on all crates to set this as a build error?

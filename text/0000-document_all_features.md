- Feature Name: document_all_features
- Start Date: 2016-06-03
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary
[summary]: #summary

One of the major goals of Rust's development process is *stability without stagnation*. That means we add features regularly. However, it can be difficult to *use* those features if they are not publicly documented anywhere. Therefore, this RFC proposes requiring that all new language features and public standard library items must be documented before landing on the stable release branch (item documentation for the standard library; in the language reference for language features).

# Motivation
[motivation]: #motivation

At present, new language features are often documented *only* in the RFCs which propose them and the associated announcement blog posts. Moreover, as features change, the existing official language documentation (the Rust Book, Rust by Example, and the language reference) can increasingly grow outdated.

Although the Rust Book and Rust by Example are kept relatively up to date, [the reference is not][home-to-reference]:

> While Rust does not have a specification, the reference tries to describe its working in detail. *It tends to be out of date.* (emphasis mine)

Importantly, though, this warning only appears on the [main site][home-to-reference], not in the reference itself. If someone searches for e.g. that `deprecated` attribute and *does* find the discussion of the deprecated attribute, they will have no reason to believe that the reference is wrong.

[home-to-reference]: https://www.rust-lang.org/documentation.html

For example, the change in Rust 1.9 to allow users to use the `#[deprecated]` attribute for their own libraries is, at the time of writing this RFC, *nowhere* reflected in official documentation. (Many other examples could be supplied; this one is chosen for its relative simplicity and recency.) The Book's [discussion of attributes][book-attributes] links to the [reference list of attributes][ref-attributes], but as of the time of writing the reference [still specifies][ref-compiler-attributes] that `deprecated` is a compiler-only feature. The two places where users might become aware of the change are [the Rust 1.9 release blog post][1.9-blog] and the [RFC itself][RFC-1270]. Neither (yet) ranks highly in search; users are likely to be misled.

[book-attributes]: https://doc.rust-lang.org/book/attributes.html
[ref-attributes]: https://doc.rust-lang.org/reference.html#attributes
[ref-compiler-attributes]: https://doc.rust-lang.org/reference.html#compiler-features
[1.9-blog]: http://blog.rust-lang.org/2016/05/26/Rust-1.9.html#deprecation-warnings
[RFC-1270]: https://github.com/rust-lang/rfcs/blob/master/text/1270-deprecation.md

Changing this to require all language features to be documented before stabilization would mean Rust users can use the language documentation with high confidence that it will provide exhaustive coverage of all stable Rust features.

Although the standard library is in excellent shape regarding documentation, including it in this policy will help guarantee that it remains so going forward.

## The Current Situation
[current-situation]: #the-current-situation

Today, the canonical source of information about new language features is the RFCs which define them.

There are several serious problems with the _status quo_:

1. Many users of Rust may simply not know that these RFCs exist. The number of users who do not know (or especially care) about the RFC process or its history will only increase as Rust becomes more popular.

2. In many cases, especially in more complicated language features, some important elements of the decision, details of implementation, and expected behavior are fleshed out either in the associated RFC (pull-request) discussion or in the implementation issues which follow them.

3. The RFCs themselves, and even more so the associated pull request discussions, are often dense with programming langauge theory. This is as it should be in context, but it means that the relevant information may be inaccessible to Rust users without prior PLT background, or without the patience to wade through it.

4. Similarly, information about the final decisions on language features is often buried deep at the end of long and winding threads (especially for a complicated feature like `impl` specialization).

5. Information on how the features will be used is often closely coupled to information on how the features will be implemented, both in the RFCs and in the discussion threads. Again, this is as it should be, but it makes it difficult (at best!) for ordinary Rust users to read.

In short, RFCs are a poor source of information about language features for the ordinary Rust user. Rust users should not need to be troubled with details of how the language is implemented works simply to learn how pieces of it work. Nor should they need to dig through tens (much less hundreds) of comments to determine what the final form of the feature is.

## Precedent
[precedent]: #precedent

This exact idea has been adopted by the Ember community after their somewhat bumpy transitions at the end of their 1.x cycle and leading into their 2.x transition. As one commenter there [put it][@davidgoli]:

> The fact that 1.13 was released without updated guides is really discouraging to me as an Ember adopter. It may be much faster, the features may be much cooler, but to me, they don't exist unless I can learn how to use them from documentation. Documentation IS feature work. ([@davidgoli])

[@davidgoli]: https://github.com/emberjs/rfcs/pull/56#issuecomment-114635962

The Ember core team agreed, and embraced the principle outlined in [this comment][guarav0]:

> No version shall be released until guides and versioned API documentation is ready. This will allow newcomers the ability to understand the latest release. ([@guarav0])

[guarav0]: https://github.com/emberjs/rfcs/pull/56#issuecomment-114339423

One of the main reasons not to adopt this approach, that it might block features from landing as soon as they otherwise might, was [addressed][@eccegordo] in that discussion as well:

> Now if this documentation effort holds up the releases people are going to grumble. But so be it. The challenge will be to effectively parcel out the effort and relieve the core team to do what they do best. No single person should be a gate. But lack of good documentation should gate releases. That way a lot of eyes are forced to focus on the problem. We can't get the great new toys unless everybody can enjoy the toys. ([@eccegordo])

[@eccegordo]: https://github.com/emberjs/rfcs/pull/56#issuecomment-114389963

The basic decision has led to a substantial improvement in the currency of the documentation (which is now updated the same day as a new version is released). Moreover, it has spurred ongoing development of better tooling around documentation to manage these releases. Finally, at least in the RFC author's estimation, it has also led to a substantial increase in the overall quality of that documentation, possibly as a consequence of increasing the community involvement in the documentation process (including the formation of a documentation subteam).

# Detailed design
[design]: #detailed-design

The basic process of developing new language features will remain unchanged from today, with the addition of a straightforward requirement that they be properly documented before being merged to stable.

## Language features
[language-features]: #language-features

In the case of language features, this will be a manual process, involving updates to the `reference.md` file. (It may at some point be sensible to break up the Reference file for easier maintenance; that is left aside as orthogonal to this discussion.)

Note that the feature documentation does not need to be written by the feature author. In fact, this is one of the areas where the community may be most able to support core developers even if not themselves programming language theorists or compiler hackers. This may free up the compiler developers' time. It will also help communicate the features in a way that is accessible to ordinary Rust users.

New features do not need to be documented to be merged into `master`/nightly, and in many cases *should* not, since the features may change substantially before landing on stable, at which point the reference material would need to be rewritten.

Instead, the documentation process should immediately precede the move to stabilize. Once the *feature* has been deemed ready for stabilization, either the author or a community volunteer should write the *reference material* for the feature.

This need not be especially long, but it should be long enough for ordinary users to learn how to use the language feature *without reading the RFCs*.

When the core team discusses whether to stabilize a feature in a given release, the reference material will now be a part of that decision. Once the feature *and* reference material are complete, it will be merged normally, and the pull request will simply include the reference material as well as the new feature.

## Standard library
[std]: #standard-library

In the case of the standard library, this could conceivably be managed by setting the `#[forbid(missing_docs)]` attribute on the library roots. In lieu of that, manual code review and general discipline should continue to serve. However, if automated tools *can* be employed here, they should.

# Drawbacks
[drawbacks]: #drawbacks

The largest drawback at present is that the language reference is *already* quite out of date. It may take substantial work to get it up to date so that new changes can be landed appropriately. (Arguably, however, this should be done regardless, since the language reference is an important part of the language ecosystem.)

Another potential issue is that some sections of the reference are particularly thorny and must be handled with considerable care (e.g. lifetimes). Although in general it would not be necessary for the author of the new language feature to write all the documentation, considerable extra care and oversight would need to be in place for these sections.

Finally, this may delay landing features on stable. However, all the points raised in [**Precedent**][precedent] on this apply, especially:

> We can't get the great new toys unless everybody can enjoy the toys. ([@eccegordo])

For Rust to attain its goal of *stability without stagnation*, its documentation must also be stable and not stagnant.

# Alternatives
[alternatives]: #alternatives

- **No change; leave RFCs as canonical documentation.**

	This approach can take (at least) two forms:

	1. We can leave things as they are, where the RFC and surrounding discussion form the primary point of documentation for newer-than-1.0 language features. As part of that, we could just link more prominently to the RFC repository and describe the process from the documentation pages.
	2. We could automatically render the text of the RFCs into part of the documentation used on the site (via submodules and the existing tooling around Markdown documents used for Rust documentation).

	However, for all the reasons highlighted above in [**Motivation: The Current Situation**][current-situation], RFCs and their associated threads are *not* a good canonical source of information on language features.

- **Add a rule for the standard library but not for language features.**
	
	This would basically just turn the _status quo_ into an official policy. It has all the same drawbacks as no change at all, but with the possible benefit of enabling automated checks on standard library documentation.

- **Add a rule for language features but not for the standard library.**
	
	The standard library is in much better shape, in no small part because of the ease of writing inline documentation for new modules. Adding a formal rule may not be necessary if good habits are already in place.

	On the other hand, having a formal policy would not seem to *hurt* anything here; it would simply formalize what is already happening (and perhaps, via linting attributes, make it easy to spot when it has failed).


# Unresolved questions
[unresolved]: #unresolved-questions

- How will the requirement for documentation in the reference be enforced?
- Given that the reference is out of date, does it need to be brought up to date before beginning enforcement of this policy?
- For the standard library, once it migrates to a crates structure, should it simply include the `#[forbid(missing_docs)]` attribute on all crates to set this as a build error?
- Is a documentation subteam, _a la_ the one used by Ember, worth creating?
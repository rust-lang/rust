// The mere existence of a user-written impl immediately disqualifies any potential
// auto trait candidates. It doesn't matter if the impl is inapplicable for the goal
// (due to unsatisfied predicates or unification errors).
//
// Consequently, we shouldn't synthesize any auto trait impls in such a case. Let's check that.

#![feature(negative_impls)] // only used in one test case
#![crate_name = "it"]

//@ has it/struct.Wrap.html
pub struct Wrap<T>(T);

// Disqualifies all auto candidates.
//
//@ has - '//*[@id="trait-implementations-list"]//*[@class="impl"]//*[@class="code-header"]' \
//        'impl<T> Unpin for Wrap<T>'
//@ !has - '//*[@id="synthetic-implementations-list"]//*[@class="impl"]//*[@class="code-header"]' \
//         'impl<T> Unpin for Wrap<T>where T: Unpin'
impl<T> Unpin for Wrap<T> {}

// Even though `<T> Wrap<T>` and `Wrap<()>` don't unify, it disqualifies all auto candidates.
//@ has - '//*[@id="trait-implementations-list"]//*[@class="impl"]//*[@class="code-header"]' \
//        'impl UnwindSafe for Wrap<()>'
//@ !has - '//*[@id="synthetic-implementations-list"]//*[@class="impl"]//*[@class="code-header"]' \
//         'impl<T> UnwindSafe for Wrap<T>where T: UnwindSafe'
impl std::panic::UnwindSafe for Wrap<()> {}

// Even though it has stricter requirements, it disqualifies all auto candidates.
//@ has - '//*[@id="trait-implementations-list"]//*[@class="impl"]//*[@class="code-header"]' \
//        'impl<T: Copy> Send for Wrap<T>'
//@ !has - '//*[@id="synthetic-implementations-list"]//*[@class="impl"]//*[@class="code-header"]' \
//         'impl<T> Send for Wrap<T>where T: Send'
unsafe impl<T: Copy> Send for Wrap<T> {}

// Disqualifies all auto candidates.
//@ has - '//*[@id="trait-implementations-list"]//*[@class="impl"]//*[@class="code-header"]' \
//        'impl<T> !Sync for Wrap<T>'
//@ !has - '//*[@id="synthetic-implementations-list"]//*[@class="impl"]//*[@class="code-header"]' \
//         'impl<T> !Sync for Wrap<T>'
impl<T> !Sync for Wrap<T> {}

// Countercheck & sanity check: `<T> Wrap<T>` does auto-impl `RefUnwindSafe` and `Freeze`.
//
// We also do this since we're using negative checks above (namely, `!has`),
// so we need to ensure that the overall output format is up to date,
// otherwise they might no longer test the intended thing.
//
// IMPORTANT: If you came here to update the rendered output of these impls,
//            you *MUST* also update all `!has` checks above accordingly!
//@ has - '//*[@id="synthetic-implementations-list"]//*[@class="impl"]//*[@class="code-header"]' \
//         'impl<T> RefUnwindSafe for Wrap<T>where T: RefUnwindSafe'
//@ has - '//*[@id="synthetic-implementations-list"]//*[@class="impl"]//*[@class="code-header"]' \
//         'impl<T> Freeze for Wrap<T>where T: Freeze'

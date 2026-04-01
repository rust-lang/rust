// Regression test for <https://github.com/rust-lang/rust/issues/148300>
//
// This ensures that extern crates in search results link to the correct url.

const EXPECTED = [
    {
        query: 'st2',
        others: [
            { name: 'st2', href: 'https://doc.rust-lang.org/nightly/std/index.html' }
        ],
    },
];

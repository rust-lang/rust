// rustfmt-normalize_comments: true
// rustfmt-wrap_comments: true

//! Doc comment
fn test() {
    //! Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec a diam
    //! lectus. Sed sit amet ipsum mauris. Maecenas congue ligula ac quam

    // comment
    // comment2

    code(); // leave this comment alone!
            // ok?

    // Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec a
    // diam lectus. Sed sit amet ipsum mauris. Maecenas congue ligula ac quam
    // viverra nec consectetur ante hendrerit. Donec et mollis dolor.
    // Praesent et diam eget libero egestas mattis sit amet vitae augue. Nam
    // tincidunt congue enim, ut porta lorem lacinia consectetur. Donec ut
    // libero sed arcu vehicula ultricies a non tortor. Lorem ipsum dolor sit
    // amet, consectetur adipiscing elit. Aenean ut gravida lorem. Ut turpis
    // felis, pulvinar a semper sed, adipiscing id dolor.

    // Very looooooooooooooooooooooooooooooooooooooooooooooooooooooooong comment
    // that should be split

    // println!("{:?}", rewrite_comment(subslice,
    //                                       false,
    //                                       comment_width,
    //                                       self.block_indent,
    //                                       self.config)
    //                           .unwrap());

    funk(); // dontchangeme
            // or me

    // #1388
    const EXCEPTION_PATHS: &'static [&'static str] = &[
        // std crates
        "src/libstd/sys/", // Platform-specific code for std lives here.
        "src/bootstrap",
    ];
}

/// test123
fn doc_comment() {}

fn chains() {
    foo.bar(|| {
        let x = 10;
        // comment
        x
    })
}

fn issue_1086() {
    //
}

// random comment

fn main() { // Test
}

// #1643
fn some_fn() // some comment
{
}

fn some_fn1()
// some comment
{
}

fn some_fn2() // some comment
{
}

fn some_fn3() // some comment some comment some comment some comment some comment some comment so
{
}

fn some_fn4()
// some comment some comment some comment some comment some comment some comment some comment
{
}

// #1603
pub enum Foo {
    A, // `/** **/`
    B, // `/*!`
    C,
}

// A test of the macro system. Can we do HTML literals?

// xfail-pretty
// xfail-test

macro_rules! html {
    { $($body:tt)* } => {
        let builder = HTMLBuilder();
        build_html!{builder := $($body)*};
        builder.getDoc()
    }
}

macro_rules! build_html {
    { $builder:expr := </$tag:ident> $($rest:tt)* } => {
        $builder.endTag(stringify!($tag));
        build_html!{ $builder := $($rest)* };
    };

    { $builder:expr := <$tag:ident> $($rest:tt)* } => {
        $builder.beginTag(stringify!($tag));
        build_html!{ $builder := $($rest)* };
    };

    { $builder:expr := . $($rest:tt)* } => {
        $builder.addText(~".");
        build_html!{ $builder := $($rest)* };
    };

    { $builder:expr := $word:ident $($rest:tt)* } => {
        $builder.addText(stringify!($word));
        build_html!{ $builder := $($rest)* };
    };

    { $builder:expr := } => { }
}

fn main() {

    let page = html! {
        <html>
            <head><title>This is the title.</title></head>
            <body>
            <p>This is some text</p>
            </body>
        </html>
    };

    // When we can do this, we are successful:
    //
    //let page = tag(~"html", ~[tag(~"head", ~[...])])

}

enum HTMLFragment {    
}

struct HTMLBuilder {
    bar: ();
    fn getDoc() -> HTMLFragment { fail }
    fn beginTag(tag: ~str) { }
    fn endTag(tag: ~str) { }
    fn addText(test: ~str) { }
}

fn HTMLBuilder() -> HTMLBuilder {
    HTMLBuilder { bar: () }
}
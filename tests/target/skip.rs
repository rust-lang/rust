// Test the skip attribute works

#[rustfmt::skip]
fn foo() { badly; formatted; stuff
; }

#[rustfmt::skip]
trait Foo
{
fn foo(
);
}

impl LateLintPass for UsedUnderscoreBinding {
    #[cfg_attr(rustfmt, rustfmt::skip)]
    fn check_expr() { // comment
    }
}

fn issue1346() {
    #[cfg_attr(rustfmt, rustfmt::skip)]
    Box::new(self.inner.call(req).then(move |result| {
        match result {
            Ok(resp) => Box::new(future::done(Ok(resp))),
            Err(e) => {
                try_error!(clo_stderr, "{}", e);
                Box::new(future::err(e))
            }
        }
    }))
}

fn skip_on_statements() {
    // Outside block
    #[rustfmt::skip]
    {
        foo; bar;
            // junk
    }

    {
        // Inside block
        #![rustfmt::skip]
        foo; bar;
            // junk
    }

    // Semi
    #[cfg_attr(rustfmt, rustfmt::skip)]
    foo(
        1, 2, 3, 4,
        1, 2,
        1, 2, 3,
    );

    // Local
    #[cfg_attr(rustfmt, rustfmt::skip)]
    let x = foo(  a,   b  ,  c);

    // Item
    #[cfg_attr(rustfmt, rustfmt::skip)]
    use foobar;

    // Mac
    #[cfg_attr(rustfmt, rustfmt::skip)]
    vec![
        1, 2, 3, 4,
        1, 2, 3, 4,
        1, 2, 3, 4,
        1, 2, 3,
        1,
        1, 2,
        1,
    ];

    // Expr
    #[cfg_attr(rustfmt, rustfmt::skip)]
    foo(  a,   b  ,  c)
}

// Check that the skip attribute applies to other attributes.
#[rustfmt::skip]
#[cfg
(  a , b
)]
fn
main() {}

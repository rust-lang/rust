// Test the skip attribute works

#[rustfmt_skip]
fn foo() { badly; formatted; stuff
; }

#[rustfmt_skip]
trait Foo
{
fn foo(
);
}

impl LateLintPass for UsedUnderscoreBinding {
    #[cfg_attr(rustfmt, rustfmt_skip)]
    fn check_expr() { // comment
    }
}

fn issue1346() {
    #[cfg_attr(rustfmt, rustfmt_skip)]
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

// run-pass
#[allow(unused)]
fn main() {
    || {
        'label: loop {
        }
    };

    // More cases added from issue 31754

    'label2: loop {
        break;
    }

    let closure = || {
        'label2: loop {}
    };

    fn inner_fn() {
        'label2: loop {}
    }
}

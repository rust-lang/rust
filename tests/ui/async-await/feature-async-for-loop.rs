// edition:2021
// gate-test-async_for_loop

#![feature(async_stream_from_iter, async_stream)]

fn f() {
    let _ = async {
        for await _i in core::stream::from_iter(0..3) {
            //~^ ERROR `for await` loops are experimental
        }
    };
}

#[cfg(FALSE)]
fn g() {
    let _ = async {
        for await _i in core::stream::from_iter(0..3) {
            //~^ ERROR `for await` loops are experimental
        }
    };
}

fn main() {}

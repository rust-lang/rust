#![warn(clippy::no_effect_underscore_binding)]
#![no_main]

trait AsyncTrait {
    async fn bar(i: u64);
}

struct Bar;

impl AsyncTrait for Bar {
    // Shouldn't lint `binding to `_` prefixed variable with no side-effect`
    async fn bar(_i: u64) {
        let _a = 0;
        //~^ no_effect_underscore_binding

        // Shouldn't lint `binding to `_` prefixed variable with no side-effect`
        let _b = num();

        let _ = async {
            let _c = 0;
            //~^ no_effect_underscore_binding

            // Shouldn't lint `binding to `_` prefixed variable with no side-effect`
            let _d = num();
        }
        .await;
    }
}

// Shouldn't lint `binding to `_` prefixed variable with no side-effect`
async fn foo(_i: u64) {
    let _a = 0;
    //~^ no_effect_underscore_binding

    // Shouldn't lint `binding to `_` prefixed variable with no side-effect`
    let _b = num();

    let _ = async {
        let _c = 0;
        //~^ no_effect_underscore_binding

        // Shouldn't lint `binding to `_` prefixed variable with no side-effect`
        let _d = num();
    }
    .await;
}

fn num() -> usize {
    0
}

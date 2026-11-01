//@ check-pass

#![deny(implicit_provenance_casts)]
//~^ WARNING unknown lint: `implicit_provenance_casts`

// feature-gating also applies when the old names are helpfully replaced with the new one:
#![deny(fuzzy_provenance_casts)]
//~^ WARNING lint `fuzzy_provenance_casts` has been renamed to `implicit_provenance_casts`
//~| WARNING unknown lint: `implicit_provenance_casts`
#![deny(lossy_provenance_casts)]
//~^ WARNING lint `lossy_provenance_casts` has been renamed to `implicit_provenance_casts`
//~| WARNING unknown lint: `implicit_provenance_casts`

fn main() {
    // no warnings emitted since the lints are not activated

    let _dangling = 16_usize as *const u8;

    let x: u8 = 37;
    let _addr: usize = &x as *const u8 as usize;
}

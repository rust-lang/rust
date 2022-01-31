// Compiler:
//
// Run-time:
//   status: 0

#![feature(arbitrary_self_types, auto_traits, core_intrinsics, lang_items, start, intrinsics)]

#![no_std]

mod intrinsics {
    extern "rust-intrinsic" {
        pub fn abort() -> !;
    }
}

/*
 * Core
 */

mod libc {
    #[link(name = "c")]
    extern "C" {
        pub fn puts(s: *const u8) -> i32;
    }
}

#[panic_handler]
fn panic_handler(_: &core::panic::PanicInfo) -> ! {
    unsafe {
        core::intrinsics::abort();
    }
}

/*
 * Code
 */

#[start]
fn main(argc: isize, _argv: *const *const u8) -> isize {
    let var = 134217856_u128;
    let var2 = 10475372733397991552_u128;
    let var3 = 193236519889708027473620326106273939584_u128;
    let var4 = 123236519889708027473620326106273939584_u128;
    let var5 = 153236519889708027473620326106273939584_u128;
    let var6 = 18446744073709551616_i128;
    let var7 = 170141183460469231731687303715884105728_u128;

    // Shifts.
    assert_eq!(var << (argc as u128 - 1), var);
    assert_eq!(var << argc as u128, 268435712);
    assert_eq!(var << (argc + 32) as u128, 1152922604118474752);
    assert_eq!(var << (argc + 48) as u128, 75557935783508361347072);
    assert_eq!(var << (argc + 60) as u128, 309485304969250248077606912);
    assert_eq!(var << (argc + 62) as u128, 1237941219877000992310427648);
    assert_eq!(var << (argc + 63) as u128, 2475882439754001984620855296);
    assert_eq!(var << (argc + 80) as u128, 324518863143436548128224745357312);

    assert_eq!(var2 << argc as u128, 20950745466795983104);
    assert_eq!(var2 << (argc as u128 - 1), var2);
    assert_eq!(var2 << (argc + 32) as u128, 89982766606709001335848566784);
    assert_eq!(var2 << (argc + 48) as u128, 5897110592337281111546171672756224);
    assert_eq!(var2 << (argc + 60) as u128, 24154564986213503432893119171609493504);
    assert_eq!(var2 << (argc + 62) as u128, 96618259944854013731572476686437974016);
    assert_eq!(var2 << (argc + 63) as u128, 193236519889708027463144953372875948032);

    assert_eq!(var3 << argc as u128, 46190672858477591483866044780779667712);
    assert_eq!(var3 << (argc as u128 - 1), var3);
    assert_eq!(var3 << (argc + 32) as u128, 21267668304951024224840338247585366016);
    assert_eq!(var3 << (argc + 48) as u128, 1335125106377253154015353231953100800);
    assert_eq!(var3 << (argc + 60) as u128, 24154564986213503432893119171609493504);
    assert_eq!(var3 << (argc + 62) as u128, 96618259944854013731572476686437974016);
    assert_eq!(var3 << (argc + 63) as u128, 193236519889708027463144953372875948032);

    assert_eq!(var >> (argc as u128 - 1), var);
    assert_eq!(var >> argc as u128, 67108928);
    assert_eq!(var >> (argc + 32) as u128, 0);
    assert_eq!(var >> (argc + 48) as u128, 0);
    assert_eq!(var >> (argc + 60) as u128, 0);
    assert_eq!(var >> (argc + 62) as u128, 0);
    assert_eq!(var >> (argc + 63) as u128, 0);

    assert_eq!(var2 >> argc as u128, 5237686366698995776);
    assert_eq!(var2 >> (argc as u128 - 1), var2);
    assert_eq!(var2 >> (argc + 32) as u128, 1219493888);
    assert_eq!(var2 >> (argc + 48) as u128, 18608);
    assert_eq!(var2 >> (argc + 60) as u128, 4);
    assert_eq!(var2 >> (argc + 62) as u128, 1);
    assert_eq!(var2 >> (argc + 63) as u128, 0);

    assert_eq!(var3 >> (argc as u128 - 1), var3);
    assert_eq!(var3 >> argc as u128, 96618259944854013736810163053136969792);
    assert_eq!(var3 >> (argc + 32) as u128, 22495691651677250335181635584);
    assert_eq!(var3 >> (argc + 48) as u128, 343257013727985387194544);
    assert_eq!(var3 >> (argc + 60) as u128, 83802981867183932420);
    assert_eq!(var3 >> (argc + 62) as u128, 20950745466795983105);
    assert_eq!(var3 >> (argc + 63) as u128, 10475372733397991552);
    assert_eq!(var3 >> (argc + 80) as u128, 79920751444992);

    assert_eq!(var6 >> argc as u128, 9223372036854775808);
    assert_eq!((var6 - 1) >> argc as u128, 9223372036854775807);
    assert_eq!(var7 >> argc as u128, 85070591730234615865843651857942052864);

    // Casts
    assert_eq!((var >> (argc + 32) as u128) as u64, 0);
    assert_eq!((var >> argc as u128) as u64, 67108928);

    // Addition.
    assert_eq!(var + argc as u128, 134217857);

    assert_eq!(var2 + argc as u128, 10475372733397991553);
    assert_eq!(var2 + (var2 + argc as u128) as u128, 20950745466795983105);

    assert_eq!(var3 + argc as u128, 193236519889708027473620326106273939585);

    // Subtraction
    assert_eq!(var - argc as u128, 134217855);

    assert_eq!(var2 - argc as u128, 10475372733397991551);

    assert_eq!(var3 - argc as u128, 193236519889708027473620326106273939583);

    // Multiplication
    assert_eq!(var * (argc + 1) as u128, 268435712);
    assert_eq!(var * (argc as u128 + var2), 1405982069077538020949770368);

    assert_eq!(var2 * (argc + 1) as u128, 20950745466795983104);
    assert_eq!(var2 * (argc as u128 + var2), 109733433903618109003204073240861360256);

    assert_eq!(var3 * argc as u128, 193236519889708027473620326106273939584);

    assert_eq!(var4 * (argc + 1) as u128, 246473039779416054947240652212547879168);

    assert_eq!(var5 * (argc + 1) as u128, 306473039779416054947240652212547879168);

    // Division.
    assert_eq!(var / (argc + 1) as u128, 67108928);
    assert_eq!(var / (argc + 2) as u128, 44739285);

    assert_eq!(var2 / (argc + 1) as u128, 5237686366698995776);
    assert_eq!(var2 / (argc + 2) as u128, 3491790911132663850);

    assert_eq!(var3 / (argc + 1) as u128, 96618259944854013736810163053136969792);
    assert_eq!(var3 / (argc + 2) as u128, 64412173296569342491206775368757979861);
    assert_eq!(var3 / (argc as u128 + var4), 1);
    assert_eq!(var3 / (argc as u128 + var2), 18446744073709551615);

    assert_eq!(var4 / (argc + 1) as u128, 61618259944854013736810163053136969792);
    assert_eq!(var4 / (argc + 2) as u128, 41078839963236009157873442035424646528);

    0
}

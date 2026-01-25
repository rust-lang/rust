//@ run-pass
//@ needs-unwind

#![feature(rust_preserve_none_cc)]

struct CrateOf<'a> {
    mcintosh: f64,
    golden_delicious: u64,
    jonagold: Option<&'a u64>,
    rome: [u64; 12],
}

#[inline(never)]
extern "rust-preserve-none" fn oven_explosion() {
    panic!("bad time");
}

#[inline(never)]
fn bite_into(yummy: u64) -> u64 {
    let did_it_actually = std::panic::catch_unwind(move || {
        oven_explosion()
    });
    assert!(did_it_actually.is_err());
    yummy - 25
}

#[inline(never)]
extern "rust-preserve-none" fn lotsa_apples(
    honeycrisp: u64,
    gala: u32,
    fuji: f64,
    granny_smith: &[u64],
    pink_lady: (),
    and_a: CrateOf<'static>,
    cosmic_crisp: u64,
    ambrosia: f64,
    winesap: &[u64],
) -> (u64, f64, u64, u64) {
    assert_eq!(honeycrisp, 220);
    assert_eq!(gala, 140);
    assert_eq!(fuji, 210.54201234);
    assert_eq!(granny_smith, &[180, 210]);
    assert_eq!(pink_lady, ());
    assert_eq!(and_a.mcintosh, 150.0);
    assert_eq!(and_a.golden_delicious, 185);
    assert_eq!(and_a.jonagold, None); // my scales can't weight these gargantuans.
    assert_eq!(and_a.rome, [180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202]);
    assert_eq!(cosmic_crisp, 270);
    assert_eq!(ambrosia, 193.1);
    assert_eq!(winesap, &[]);
    (
        and_a.rome.iter().sum(),
        fuji + ambrosia,
        cosmic_crisp - honeycrisp,
        bite_into(and_a.golden_delicious)
    )
}

fn main() {
    let pie = lotsa_apples(220, 140, 210.54201234, &[180, 210], (), CrateOf {
        mcintosh: 150.0,
        golden_delicious: 185,
        jonagold: None,
        rome: [180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202]
    }, 270, 193.1, &[]);
    assert_eq!(pie, (2292, 403.64201234, 50, 160));
}

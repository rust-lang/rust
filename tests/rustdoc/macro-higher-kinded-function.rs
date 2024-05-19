#![crate_name = "foo"]

pub struct TyCtxt<'tcx>(&'tcx u8);

macro_rules! gen {
    ($(($name:ident, $tcx:lifetime, [$k:ty], [$r:ty]))*) => {
        pub struct Providers {
            $(pub $name: for<$tcx> fn(TyCtxt<$tcx>, $k) -> $r,)*
        }
    }
}

// @has 'foo/struct.Providers.html'
// @has - '//*[@class="rust item-decl"]//code' "pub a: for<'tcx> fn(_: TyCtxt<'tcx>, _: u8) -> i8,"
// @has - '//*[@class="rust item-decl"]//code' "pub b: for<'tcx> fn(_: TyCtxt<'tcx>, _: u16) -> i16,"
// @has - '//*[@id="structfield.a"]/code' "a: for<'tcx> fn(_: TyCtxt<'tcx>, _: u8) -> i8"
// @has - '//*[@id="structfield.b"]/code' "b: for<'tcx> fn(_: TyCtxt<'tcx>, _: u16) -> i16"
gen! {
    (a, 'tcx, [u8], [i8])
    (b, 'tcx, [u16], [i16])
}

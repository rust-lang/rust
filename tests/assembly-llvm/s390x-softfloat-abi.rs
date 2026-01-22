//@ add-minicore
//@ revisions: enable-softfloat disable-softfloat
//@ assembly-output: emit-asm
//@ compile-flags: -Copt-level=3 --crate-type=lib
//@[enable-softfloat] compile-flags: --target=s390x-unknown-none-softfloat
//@[enable-softfloat] needs-llvm-components: systemz
//@[disable-softfloat] compile-flags: --target=s390x-unknown-linux-gnu
//@[disable-softfloat] needs-llvm-components: systemz

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]

extern crate minicore;
use minicore::*;

extern "C" {
    fn extern_func(value: f64) -> f64;
}

// CHECK-LABEL: test_softfloat
#[no_mangle]
extern "C" fn test_softfloat() -> f64 {
    let value = 3.141_f64;

    // without softfloat we load the value direct to the first float register
    // we do NOT construct a softfloat in r2 (first non-float arg register)
    // disable-softfloat: ld %f{{.*}}, 0(%r{{.*}})
    // disable-softfloat-NOT: llihf %r{{.*}}, 1074340036
    // disable-softfloat-NOT: oilf %r{{.*}}, 2611340116

    // with softfloat we construct the softfloat arg in r2
    // we do NOT pass anything by f0 (first float arg register)
    // float registers can not be accessed
    // enable-softfloat: llihf %r{{.*}}, 1074340036
    // enable-softfloat-NOT: ld %f{{.*}}, 0(%r{{.*}})
    // enable-softfloat-NEXT: oilf %r{{.*}}, 2611340116

    unsafe { extern_func(value) };
    // disable-softfloat-NEXT: brasl %r{{.*}}, extern_func@PLT
    // enable-softfloat-NEXT: brasl %r{{.*}}, extern_func@PLT

    // for return we check that without softfloat we write to float register
    // disable-softfloat: ld %f{{.*}}, 0(%r{{.*}})
    // disable-softfloat-NOT: llihf %r{{.*}}, 1072841097
    // disable-softfloat-NOT: oilf %r{{.*}}, 927712936

    #[cfg(not(target_feature = "soft-float"))]
    {
        1.141_f64
    }

    // for return we check that WITH softfloat we write to genral purpose register
    // enable-softfloat: llihf %r{{.*}}, 1072841097
    // enable-softfloat-NEXT: oilf %r{{.*}}, 927712936
    // enable-softfloat-NOT: ld %f{{.*}}, 0(%r{{.*}})
    #[cfg(target_feature = "soft-float")]
    {
        2.718_f64
    }
    // enable-softfloat: br %r{{.*}}
    // disable-softfloat: br %r{{.*}}
}

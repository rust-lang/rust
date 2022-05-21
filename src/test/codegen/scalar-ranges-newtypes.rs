#![crate_type = "lib"]
#![feature(rustc_attrs, bench_black_box)]

#[rustc_layout_scalar_valid_range_start(1)]
pub struct NonNull1(*const ());

#[rustc_layout_scalar_valid_range_start(1)]
pub struct NonNull2 {
    ptr: *const (),
}

#[rustc_layout_scalar_valid_range_start(1)]
pub struct NonNull3 {
    _marker: std::marker::PhantomData<()>,
    ptr: *const (),
}

// CHECK: define void @test_nonnull_load
#[no_mangle]
pub fn test_nonnull_load(p1: &NonNull1, p2: &NonNull2, p3: &NonNull3) {
    // CHECK: %[[P1:[0-9]+]] = bitcast i8** %p1 to {}**
    // CHECK: load {}*, {}** %[[P1]], align 8, !nonnull
    std::hint::black_box(p1.0);

    // CHECK: %[[P2:[0-9]+]] = bitcast i8** %p2 to {}**
    // CHECK: load {}*, {}** %[[P2]], align 8, !nonnull
    std::hint::black_box(p2.ptr);

    // CHECK: %[[P3:[0-9]+]] = bitcast i8** %p3 to {}**
    // CHECK: load {}*, {}** %[[P3]], align 8, !nonnull
    std::hint::black_box(p3.ptr);
}

#[rustc_layout_scalar_valid_range_start(16)]
#[rustc_layout_scalar_valid_range_end(2032)]
pub struct Range(i32);

// CHECK: define void @test_range_load
#[no_mangle]
pub fn test_range_load(p: &Range) {
    // CHECK: load i32, i32* %{{.*}}, align 4, !range ![[RANGE:[0-9]+]]
    std::hint::black_box(p.0);
}

// CHECK: ![[RANGE]] = !{i32 16, i32 2033}

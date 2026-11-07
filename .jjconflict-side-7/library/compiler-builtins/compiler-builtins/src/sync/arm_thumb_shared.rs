// Used by both arm_linux.rs and thumbv6k.rs.

// References:
// - https://llvm.org/docs/Atomics.html#libcalls-sync
// - https://gcc.gnu.org/onlinedocs/gcc/_005f_005fsync-Builtins.html
// - https://refspecs.linuxfoundation.org/elf/IA64-SysV-psABI.pdf#page=58

atomic_rmw!(@old __sync_fetch_and_add_1, u8, |a: u8, b: u8| a.wrapping_add(b));
atomic_rmw!(@old __sync_fetch_and_add_2, u16, |a: u16, b: u16| a
    .wrapping_add(b));
atomic_rmw!(@old __sync_fetch_and_add_4, u32, |a: u32, b: u32| a
    .wrapping_add(b));

atomic_rmw!(@new __sync_add_and_fetch_1, u8, |a: u8, b: u8| a.wrapping_add(b));
atomic_rmw!(@new __sync_add_and_fetch_2, u16, |a: u16, b: u16| a
    .wrapping_add(b));
atomic_rmw!(@new __sync_add_and_fetch_4, u32, |a: u32, b: u32| a
    .wrapping_add(b));

atomic_rmw!(@old __sync_fetch_and_sub_1, u8, |a: u8, b: u8| a.wrapping_sub(b));
atomic_rmw!(@old __sync_fetch_and_sub_2, u16, |a: u16, b: u16| a
    .wrapping_sub(b));
atomic_rmw!(@old __sync_fetch_and_sub_4, u32, |a: u32, b: u32| a
    .wrapping_sub(b));

atomic_rmw!(@new __sync_sub_and_fetch_1, u8, |a: u8, b: u8| a.wrapping_sub(b));
atomic_rmw!(@new __sync_sub_and_fetch_2, u16, |a: u16, b: u16| a
    .wrapping_sub(b));
atomic_rmw!(@new __sync_sub_and_fetch_4, u32, |a: u32, b: u32| a
    .wrapping_sub(b));

atomic_rmw!(@old __sync_fetch_and_and_1, u8, |a: u8, b: u8| a & b);
atomic_rmw!(@old __sync_fetch_and_and_2, u16, |a: u16, b: u16| a & b);
atomic_rmw!(@old __sync_fetch_and_and_4, u32, |a: u32, b: u32| a & b);

atomic_rmw!(@new __sync_and_and_fetch_1, u8, |a: u8, b: u8| a & b);
atomic_rmw!(@new __sync_and_and_fetch_2, u16, |a: u16, b: u16| a & b);
atomic_rmw!(@new __sync_and_and_fetch_4, u32, |a: u32, b: u32| a & b);

atomic_rmw!(@old __sync_fetch_and_or_1, u8, |a: u8, b: u8| a | b);
atomic_rmw!(@old __sync_fetch_and_or_2, u16, |a: u16, b: u16| a | b);
atomic_rmw!(@old __sync_fetch_and_or_4, u32, |a: u32, b: u32| a | b);

atomic_rmw!(@new __sync_or_and_fetch_1, u8, |a: u8, b: u8| a | b);
atomic_rmw!(@new __sync_or_and_fetch_2, u16, |a: u16, b: u16| a | b);
atomic_rmw!(@new __sync_or_and_fetch_4, u32, |a: u32, b: u32| a | b);

atomic_rmw!(@old __sync_fetch_and_xor_1, u8, |a: u8, b: u8| a ^ b);
atomic_rmw!(@old __sync_fetch_and_xor_2, u16, |a: u16, b: u16| a ^ b);
atomic_rmw!(@old __sync_fetch_and_xor_4, u32, |a: u32, b: u32| a ^ b);

atomic_rmw!(@new __sync_xor_and_fetch_1, u8, |a: u8, b: u8| a ^ b);
atomic_rmw!(@new __sync_xor_and_fetch_2, u16, |a: u16, b: u16| a ^ b);
atomic_rmw!(@new __sync_xor_and_fetch_4, u32, |a: u32, b: u32| a ^ b);

atomic_rmw!(@old __sync_fetch_and_nand_1, u8, |a: u8, b: u8| !(a & b));
atomic_rmw!(@old __sync_fetch_and_nand_2, u16, |a: u16, b: u16| !(a & b));
atomic_rmw!(@old __sync_fetch_and_nand_4, u32, |a: u32, b: u32| !(a & b));

atomic_rmw!(@new __sync_nand_and_fetch_1, u8, |a: u8, b: u8| !(a & b));
atomic_rmw!(@new __sync_nand_and_fetch_2, u16, |a: u16, b: u16| !(a & b));
atomic_rmw!(@new __sync_nand_and_fetch_4, u32, |a: u32, b: u32| !(a & b));

atomic_rmw!(@old __sync_fetch_and_max_1, i8, |a: i8, b: i8| if a > b {
    a
} else {
    b
});
atomic_rmw!(@old __sync_fetch_and_max_2, i16, |a: i16, b: i16| if a > b {
    a
} else {
    b
});
atomic_rmw!(@old __sync_fetch_and_max_4, i32, |a: i32, b: i32| if a > b {
    a
} else {
    b
});

atomic_rmw!(@old __sync_fetch_and_umax_1, u8, |a: u8, b: u8| if a > b {
    a
} else {
    b
});
atomic_rmw!(@old __sync_fetch_and_umax_2, u16, |a: u16, b: u16| if a > b {
    a
} else {
    b
});
atomic_rmw!(@old __sync_fetch_and_umax_4, u32, |a: u32, b: u32| if a > b {
    a
} else {
    b
});

atomic_rmw!(@old __sync_fetch_and_min_1, i8, |a: i8, b: i8| if a < b {
    a
} else {
    b
});
atomic_rmw!(@old __sync_fetch_and_min_2, i16, |a: i16, b: i16| if a < b {
    a
} else {
    b
});
atomic_rmw!(@old __sync_fetch_and_min_4, i32, |a: i32, b: i32| if a < b {
    a
} else {
    b
});

atomic_rmw!(@old __sync_fetch_and_umin_1, u8, |a: u8, b: u8| if a < b {
    a
} else {
    b
});
atomic_rmw!(@old __sync_fetch_and_umin_2, u16, |a: u16, b: u16| if a < b {
    a
} else {
    b
});
atomic_rmw!(@old __sync_fetch_and_umin_4, u32, |a: u32, b: u32| if a < b {
    a
} else {
    b
});

atomic_rmw!(@old __sync_lock_test_and_set_1, u8, |_: u8, b: u8| b);
atomic_rmw!(@old __sync_lock_test_and_set_2, u16, |_: u16, b: u16| b);
atomic_rmw!(@old __sync_lock_test_and_set_4, u32, |_: u32, b: u32| b);

atomic_cmpxchg!(__sync_val_compare_and_swap_1, u8);
atomic_cmpxchg!(__sync_val_compare_and_swap_2, u16);
atomic_cmpxchg!(__sync_val_compare_and_swap_4, u32);

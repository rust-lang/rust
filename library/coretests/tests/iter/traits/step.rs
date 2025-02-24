use core::iter::*;

#[test]
fn test_steps_between() {
    assert_eq!(Step::steps_between(&20_u8, &200_u8), (180_usize, Some(180_usize)));
    assert_eq!(Step::steps_between(&-20_i8, &80_i8), (100_usize, Some(100_usize)));
    assert_eq!(Step::steps_between(&-120_i8, &80_i8), (200_usize, Some(200_usize)));
    assert_eq!(
        Step::steps_between(&20_u32, &4_000_100_u32),
        (4_000_080_usize, Some(4_000_080_usize))
    );
    assert_eq!(Step::steps_between(&-20_i32, &80_i32), (100_usize, Some(100_usize)));
    assert_eq!(
        Step::steps_between(&-2_000_030_i32, &2_000_050_i32),
        (4_000_080_usize, Some(4_000_080_usize))
    );

    // Skip u64/i64 to avoid differences with 32-bit vs 64-bit platforms

    assert_eq!(Step::steps_between(&20_u128, &200_u128), (180_usize, Some(180_usize)));
    assert_eq!(Step::steps_between(&-20_i128, &80_i128), (100_usize, Some(100_usize)));
    if cfg!(target_pointer_width = "64") {
        assert_eq!(
            Step::steps_between(&10_u128, &0x1_0000_0000_0000_0009_u128),
            (usize::MAX, Some(usize::MAX))
        );
    }
    assert_eq!(Step::steps_between(&10_u128, &0x1_0000_0000_0000_000a_u128), (usize::MAX, None));
    assert_eq!(Step::steps_between(&10_i128, &0x1_0000_0000_0000_000a_i128), (usize::MAX, None));
    assert_eq!(
        Step::steps_between(&-0x1_0000_0000_0000_0000_i128, &0x1_0000_0000_0000_0000_i128,),
        (usize::MAX, None),
    );

    assert_eq!(Step::steps_between(&100_u32, &10_u32), (0, None));
}

#[test]
fn test_step_forward() {
    assert_eq!(Step::forward_checked(55_u8, 200_usize), Some(255_u8));
    assert_eq!(Step::forward_checked(252_u8, 200_usize), None);
    assert_eq!(Step::forward_checked(0_u8, 256_usize), None);
    assert_eq!(Step::forward_checked(-110_i8, 200_usize), Some(90_i8));
    assert_eq!(Step::forward_checked(-110_i8, 248_usize), None);
    assert_eq!(Step::forward_checked(-126_i8, 256_usize), None);

    assert_eq!(Step::forward_checked(35_u16, 100_usize), Some(135_u16));
    assert_eq!(Step::forward_checked(35_u16, 65500_usize), Some(u16::MAX));
    assert_eq!(Step::forward_checked(36_u16, 65500_usize), None);
    assert_eq!(Step::forward_checked(-110_i16, 200_usize), Some(90_i16));
    assert_eq!(Step::forward_checked(-20_030_i16, 50_050_usize), Some(30_020_i16));
    assert_eq!(Step::forward_checked(-10_i16, 40_000_usize), None);
    assert_eq!(Step::forward_checked(-10_i16, 70_000_usize), None);

    assert_eq!(Step::forward_checked(10_u128, 70_000_usize), Some(70_010_u128));
    assert_eq!(Step::forward_checked(10_i128, 70_030_usize), Some(70_040_i128));
    assert_eq!(
        Step::forward_checked(0xffff_ffff_ffff_ffff__ffff_ffff_ffff_ff00_u128, 0xff_usize),
        Some(u128::MAX),
    );
    assert_eq!(
        Step::forward_checked(0xffff_ffff_ffff_ffff__ffff_ffff_ffff_ff00_u128, 0x100_usize),
        None
    );
    assert_eq!(
        Step::forward_checked(0x7fff_ffff_ffff_ffff__ffff_ffff_ffff_ff00_i128, 0xff_usize),
        Some(i128::MAX),
    );
    assert_eq!(
        Step::forward_checked(0x7fff_ffff_ffff_ffff__ffff_ffff_ffff_ff00_i128, 0x100_usize),
        None
    );
}

#[test]
fn test_step_backward() {
    assert_eq!(Step::backward_checked(255_u8, 200_usize), Some(55_u8));
    assert_eq!(Step::backward_checked(100_u8, 200_usize), None);
    assert_eq!(Step::backward_checked(255_u8, 256_usize), None);
    assert_eq!(Step::backward_checked(90_i8, 200_usize), Some(-110_i8));
    assert_eq!(Step::backward_checked(110_i8, 248_usize), None);
    assert_eq!(Step::backward_checked(127_i8, 256_usize), None);

    assert_eq!(Step::backward_checked(135_u16, 100_usize), Some(35_u16));
    assert_eq!(Step::backward_checked(u16::MAX, 65500_usize), Some(35_u16));
    assert_eq!(Step::backward_checked(10_u16, 11_usize), None);
    assert_eq!(Step::backward_checked(90_i16, 200_usize), Some(-110_i16));
    assert_eq!(Step::backward_checked(30_020_i16, 50_050_usize), Some(-20_030_i16));
    assert_eq!(Step::backward_checked(-10_i16, 40_000_usize), None);
    assert_eq!(Step::backward_checked(-10_i16, 70_000_usize), None);

    assert_eq!(Step::backward_checked(70_010_u128, 70_000_usize), Some(10_u128));
    assert_eq!(Step::backward_checked(70_020_i128, 70_030_usize), Some(-10_i128));
    assert_eq!(Step::backward_checked(10_u128, 7_usize), Some(3_u128));
    assert_eq!(Step::backward_checked(10_u128, 11_usize), None);
    assert_eq!(
        Step::backward_checked(-0x7fff_ffff_ffff_ffff__ffff_ffff_ffff_ff00_i128, 0x100_usize),
        Some(i128::MIN)
    );
}

import std::unsafe;

#[test]
fn reinterpret_cast() unsafe {
    assert unsafe::reinterpret_cast(1) == 1u;
}

#[test]
#[should_fail]
#[ignore(cfg(target_os = "win32"))]
fn reinterpret_cast_wrong_size() unsafe {
    let _i: uint = unsafe::reinterpret_cast(0u8);
}
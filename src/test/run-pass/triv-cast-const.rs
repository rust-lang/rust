import core::mtypes::m_int;

// This will be more interesting once there is support
// for consts that refer to other consts, i.e. math_f64::consts::pi as m_float
#[cfg(target_arch="x86")]
const foo: m_int = 0i32 as m_int;

#[cfg(target_arch="x86_64")]
const foo: m_int = 0i64 as m_int;

fn main() {
    assert foo == 0 as m_int;
}
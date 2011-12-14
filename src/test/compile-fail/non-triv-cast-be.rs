// error-pattern: non-trivial cast of tail-call return value
import core::mtypes::*;

fn foo_float() -> m_float { ret 0.0 as m_float; }
fn bar_float() -> bool { be foo_float() as bool; }

fn foo_int() -> m_int { ret 0 as m_int; }
fn bar_int() -> bool { be foo_int() as bool; }

fn foo_uint() -> m_uint { ret 0u as m_uint; }
fn bar_uint() -> bool { be foo_uint() as bool; }

fn main() {
    assert bar_float() == 0.0;
    assert bar_int() == 0.0;
    assert bar_uint() == 0.0;
}
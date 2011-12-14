use std;

import ctypes::*;

fn foo_float() -> m_float { ret 0.0 as m_float; }
fn bar_float() -> float { be foo_float() as float; }

fn foo_int() -> m_int { ret 0 as m_int; }
fn bar_int() -> int { be foo_int() as int; }

fn foo_uint() -> m_uint { ret 0u as m_uint; }
fn bar_uint() -> uint { be foo_uint() as uint; }

fn foo_long() -> long { ret 0 as long; }
fn bar_long() -> int { be foo_long() as int; }

fn foo_ulong() -> ulong { ret 0u as ulong; }
fn bar_ulong() -> uint { be foo_uint() as uint; }

fn main() {
    assert bar_float() == 0.0;
    assert bar_int() == 0;
    assert bar_uint() == 0u;
    assert bar_long() == 0;
    assert bar_ulong() == 0u;
}
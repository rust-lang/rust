//@ run-pass
//@ edition:2021

const PATTERN_REF: &str = "Hello World";
const NUMBER_POINTER: *const i32 = 30 as *const i32;

pub fn edge_case_ref(event: &str) {
    let _ = || {
        match event {
            PATTERN_REF => (),
            _ => (),
        };
    };
}

pub fn edge_case_str(event: String) {
    let _ = || {
        match event.as_str() {
            "hello" => (),
            _ => (),
        };
    };
}

pub fn edge_case_raw_ptr(event: *const i32) {
    let _ = || {
        match event {
            NUMBER_POINTER => (),
            _ => (),
        };
    };
}

pub fn edge_case_char(event: char) {
    let _ = || {
        match event {
            'a' => (),
            _ => (),
        };
    };
}

fn main() {}


use std;
import std::option::*;

fn baz() -> ! { fail; }

fn foo() {
    alt (some[int](5)) {
        case (some[int](?x)) {
            auto bar;
            alt (none[int]) {
                case (none[int]) { bar = 5; }
                case (_) { baz(); }
            }
            log bar;
        }
        case (none[int]) { log "hello"; }
    }
}

fn main() { foo(); }
// error-pattern:explicit failure
// Don't double free the string
use std;
import io::{reader, reader_util};

fn main() {
    do io::with_str_reader("") |rdr| {
        alt rdr.read_char() { '=' { } _ { fail } }
    }
}

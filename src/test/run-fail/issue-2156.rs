// error-pattern:explicit failure
// Don't double free the string
use std;
import io::reader;

fn main() {
    do io::with_str_reader(~"") |rdr| {
        match rdr.read_char() { '=' => { } _ => { fail } }
    }
}

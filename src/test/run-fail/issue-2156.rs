// error-pattern:explicit failure
// Don't double free the string
extern mod std;
use io::Reader;

fn main() {
    do io::with_str_reader(~"") |rdr| {
        match rdr.read_char() { '=' => { } _ => { fail } }
    }
}

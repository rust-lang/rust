extern mod std;
use io::WriterUtil;
use std::tempfile;

fn main() {
    let dir = option::unwrap(tempfile::mkdtemp(&Path("."), ""));
    let path = dir.with_filename("file");

    {
        match io::file_writer(&path, [io::Create, io::Truncate]) {
            Err(copy e) => fail e,
            Ok(f) => {
                for uint::range(0, 1000) |_i| {
                    f.write_u8(0);
                }
            }
        }
    }

    assert path.exists();
    assert path.get_size() == Some(1000);

    os::remove_dir(&dir);
}

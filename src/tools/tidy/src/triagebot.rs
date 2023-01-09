use std::path::Path;
use toml::Value;


pub fn check(path: &Path, bad: &mut bool) {
        let filename = path.file_name().unwrap();
        if filename != "triagebot.toml" {
            return;
        }
        let contents = std::fs::read_to_string(filename).unwrap();

        let conf = contents.parse::<Value>();
        match conf {
            Ok(_) => {}
            Err(_err) => {
                tidy_error!(bad, "triagebot.toml does not have valid TOML format")
            }
        }
}

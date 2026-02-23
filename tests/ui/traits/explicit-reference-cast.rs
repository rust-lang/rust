// compile-fail

use std::convert::TryFrom;
use std::path::{Path, PathBuf};

pub struct ToolA(PathBuf);
//~^ HELP the trait `From<&PathBuf>` is not implemented for `ToolA`

impl From<&Path> for ToolA {
    //~^ HELP the following other types implement trait `From<T>`
    fn from(p: &Path) -> ToolA {
        ToolA(p.to_path_buf())
    }
}

// Add a different From<T> impl to ensure we suggest the correct cast
impl From<&str> for ToolA {
    fn from(s: &str) -> ToolA {
        ToolA(PathBuf::from(s))
    }
}

pub struct ToolB(PathBuf);
//~^ HELP the trait `From<&PathBuf>` is not implemented for `ToolB`
//~| HELP the trait `From<&PathBuf>` is not implemented for `ToolB`

impl TryFrom<&Path> for ToolB {
    //~^ HELP the trait `TryFrom<&PathBuf>` is not implemented for `ToolB`
    //~| HELP the trait `TryFrom<&PathBuf>` is not implemented for `ToolB`
    type Error = ();

    fn try_from(p: &Path) -> Result<ToolB, ()> {
        Ok(ToolB(p.to_path_buf()))
    }
}

fn main() {
    let path = PathBuf::new();

    let _ = ToolA::from(&path);
    //~^ ERROR the trait bound `ToolA: From<&PathBuf>` is not satisfied
    //~| HELP consider casting the `&PathBuf` value to `&Path`
    let _ = ToolB::try_from(&path);
    //~^ ERROR the trait bound `ToolB: TryFrom<&PathBuf>` is not satisfied
    //~| ERROR the trait bound `ToolB: From<&PathBuf>` is not satisfied
    //~| HELP consider casting the `&PathBuf` value to `&Path`
    //~| HELP consider casting the `&PathBuf` value to `&Path`
    //~| HELP for that trait implementation, expected `Path`, found `PathBuf`
    //~| HELP for that trait implementation, expected `Path`, found `PathBuf`
}

//! A repository of supported licences

/// Fetch a license text
pub fn get(name: &str) -> Result<&'static str, anyhow::Error> {
    if name == "0BSD" {
        Ok(include_str!("../licenses/0BSD.txt"))
    } else if name == "Apache-2.0" {
        Ok(include_str!("../licenses/Apache-2.0.txt"))
    } else if name == "BSD-1-Clause" {
        Ok(include_str!("../licenses/BSD-1-Clause.txt"))
    } else if name == "BSD-2-Clause" {
        Ok(include_str!("../licenses/BSD-2-Clause.txt"))
    } else if name == "BSD-3-Clause" {
        Ok(include_str!("../licenses/BSD-3-Clause.txt"))
    } else if name == "BSL-1.0" {
        Ok(include_str!("../licenses/BSL-1.0.txt"))
    } else if name == "CC-BY-SA-4.0" {
        Ok(include_str!("../licenses/CC-BY-SA-4.0.txt"))
    } else if name == "CC0-1.0" {
        Ok(include_str!("../licenses/CC0-1.0.txt"))
    } else if name == "ISC" {
        Ok(include_str!("../licenses/ISC.txt"))
    } else if name == "LGPL-2.1-or-later" {
        Ok(include_str!("../licenses/LGPL-2.1-or-later.txt"))
    } else if name == "LLVM-exception" {
        Ok(include_str!("../licenses/LLVM-exception.txt"))
    } else if name == "MIT" {
        Ok(include_str!("../licenses/MIT.txt"))
    } else if name == "MIT-0" {
        Ok(include_str!("../licenses/MIT-0.txt"))
    } else if name == "MPL-2.0" {
        Ok(include_str!("../licenses/MPL-2.0.txt"))
    } else if name == "NCSA" {
        Ok(include_str!("../licenses/NCSA.txt"))
    } else if name == "OFL-1.1" {
        Ok(include_str!("../licenses/OFL-1.1.txt"))
    } else if name == "Unicode-3.0" {
        Ok(include_str!("../licenses/Unicode-3.0.txt"))
    } else if name == "Unicode-DFS-2016" {
        Ok(include_str!("../licenses/Unicode-DFS-2016.txt"))
    } else if name == "Unlicense" {
        Ok(include_str!("../licenses/Unlicense.txt"))
    } else if name == "Zlib" {
        Ok(include_str!("../licenses/Zlib.txt"))
    } else {
        Err(anyhow::format_err!("Could not find license file {}", name))
    }
}

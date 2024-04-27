#[cfg(test)]
mod tests;

pub fn to_readable_str(mut val: usize) -> String {
    let mut groups = vec![];
    loop {
        let group = val % 1000;

        val /= 1000;

        if val == 0 {
            groups.push(group.to_string());
            break;
        } else {
            groups.push(format!("{group:03}"));
        }
    }

    groups.reverse();

    groups.join("_")
}

// const wrapper for `if let Some((_, tail)) = name.rsplit_once(':') { tail } else { name }`
pub const fn c_name(name: &'static str) -> &'static str {
    // FIXME Simplify the implementation once more `str` methods get const-stable.
    // and inline into call site
    let bytes = name.as_bytes();
    let mut i = bytes.len();
    while i > 0 && bytes[i - 1] != b':' {
        i = i - 1;
    }
    let (_, bytes) = bytes.split_at(i);
    match std::str::from_utf8(bytes) {
        Ok(name) => name,
        Err(_) => name,
    }
}

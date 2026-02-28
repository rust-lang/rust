use std::path::Path;

pub fn normalize_abs(path: &Path) -> String {
    match path.canonicalize() {
        Ok(canon) => canon.to_string_lossy().into_owned(),
        Err(_) => path.to_string_lossy().into_owned(),
    }
}

pub fn relativize(path: &Path, base: &Path) -> String {
    if let Ok(rel) = path.strip_prefix(base) {
        rel.to_string_lossy().into_owned()
    } else {
        path.to_string_lossy().into_owned()
    }
}

pub fn redact_run_root(text: &str, run_root: &Path) -> String {
    let root = run_root.to_string_lossy();
    text.replace(root.as_ref(), "<run-root>")
}

pub fn artifact_kind(path: &Path) -> String {
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or_default();
    match ext {
        "rmeta" => "rmeta",
        "rlib" => "rlib",
        "so" | "dylib" | "dll" => "dylib",
        "a" => "staticlib",
        "o" | "obj" => "obj",
        "ll" => "llvm-ir",
        "bc" => "llvm-bc",
        "d" => "dep-info",
        "html" => "rustdoc-html",
        "json" => "json",
        "txt" | "md" | "toml" => "text",
        _ => {
            if path.file_name().and_then(|n| n.to_str()).map_or(false, |n| n.contains(".rmeta")) {
                "rmeta"
            } else {
                "binary"
            }
        }
    }
    .to_string()
}

pub fn wildcard_match(pattern: &str, candidate: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    let mut p_iter = pattern.split('*').peekable();
    let mut rest = candidate;
    let mut is_first = true;

    while let Some(part) = p_iter.next() {
        if part.is_empty() {
            continue;
        }

        if is_first && !pattern.starts_with('*') {
            if !rest.starts_with(part) {
                return false;
            }
            rest = &rest[part.len()..];
        } else if let Some(idx) = rest.find(part) {
            rest = &rest[idx + part.len()..];
        } else {
            return false;
        }

        if p_iter.peek().is_none() && !pattern.ends_with('*') && !rest.is_empty() {
            return false;
        }
        is_first = false;
    }

    true
}

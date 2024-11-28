use cargo_metadata::Metadata;
use clippy_utils::diagnostics::span_lint;
use rustc_lint::LateContext;
use rustc_span::DUMMY_SP;

use super::CARGO_COMMON_METADATA;

pub(super) fn check(cx: &LateContext<'_>, metadata: &Metadata, ignore_publish: bool) {
    for package in &metadata.packages {
        // only run the lint if publish is `None` (`publish = true` or skipped entirely)
        // or if the vector isn't empty (`publish = ["something"]`)
        if package.publish.as_ref().filter(|publish| publish.is_empty()).is_none() || ignore_publish {
            if is_empty_str(package.description.as_ref()) {
                missing_warning(cx, package, "package.description");
            }

            if is_empty_str(package.license.as_ref()) && is_empty_str(package.license_file.as_ref()) {
                missing_warning(cx, package, "either package.license or package.license_file");
            }

            if is_empty_str(package.repository.as_ref()) {
                missing_warning(cx, package, "package.repository");
            }

            if is_empty_str(package.readme.as_ref()) {
                missing_warning(cx, package, "package.readme");
            }

            if is_empty_vec(package.keywords.as_ref()) {
                missing_warning(cx, package, "package.keywords");
            }

            if is_empty_vec(package.categories.as_ref()) {
                missing_warning(cx, package, "package.categories");
            }
        }
    }
}

fn missing_warning(cx: &LateContext<'_>, package: &cargo_metadata::Package, field: &str) {
    let message = format!("package `{}` is missing `{field}` metadata", package.name);
    span_lint(cx, CARGO_COMMON_METADATA, DUMMY_SP, message);
}

fn is_empty_str<T: AsRef<std::ffi::OsStr>>(value: Option<&T>) -> bool {
    value.is_none_or(|s| s.as_ref().is_empty())
}

fn is_empty_vec(value: &[String]) -> bool {
    // This works because empty iterators return true
    value.iter().all(String::is_empty)
}

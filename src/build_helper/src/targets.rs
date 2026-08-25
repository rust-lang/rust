// FIXME(#142296): this hack is because there is no reliable way (yet) to determine whether a given
// target supports std. In the long-term, we should try to implement a way to *reliably* determine
// target (std) metadata.
//
// NOTE: this is pulled out to `build_helpers` to share this hack between `bootstrap` and
// `compiletest`.
pub fn target_supports_std(target_tuple: &str) -> bool {
    !(target_tuple.contains("-none")
        || target_tuple.contains("nvptx")
        || target_tuple.contains("switch"))
}

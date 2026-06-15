fn main() {
    // None of these shall be allowed without the feature gate.
    let _ = std::io::OnBrokenPipe::BackwardsCompatible; //~ ERROR: use of unstable library feature `on_broken_pipe`
    let _ = std::io::OnBrokenPipe::Kill; //~ ERROR: use of unstable library feature `on_broken_pipe`
    let _ = std::io::OnBrokenPipe::Error; //~ ERROR: use of unstable library feature `on_broken_pipe`
    let _ = std::io::OnBrokenPipe::Inherit; //~ ERROR: use of unstable library feature `on_broken_pipe`
    let _ = std::io::on_broken_pipe(); //~ ERROR: use of unstable library feature `on_broken_pipe`
}

use super::*;

#[test]
fn test_span_hash_one_line() {
    let source = "some text\ntidy-tag\ncheckme=42\ntidy-tag\n";
    let tag = "tidy-tag";
    assert_eq!("42258eba764c3f94a24de379e5715dc8", span_hash(source, tag, &mut true).unwrap());
}

#[test]
fn test_span_hash_multiple_lines() {
    let source = "some text\ntidy-tag\ncheckme=42\nother line\ntidy-tag\n";
    let tag = "tidy-tag";
    assert_eq!("49cb23dc2032ceea671ca48092750a1c", span_hash(source, tag, &mut true).unwrap());
}

#[test]
fn test_span_hash_has_some_text_in_line_with_tag() {
    let source = "some text\ntidy-tag ignore me\ncheckme=42\nother line\ntidy-tag\n";
    let tag = "tidy-tag";
    assert_eq!("49cb23dc2032ceea671ca48092750a1c", span_hash(source, tag, &mut true).unwrap());
}

#[test]
fn test_span_hash_has_some_text_in_line_before_second_tag() {
    let source = r#"
RUN ./build-clang.sh
ENV CC=clang CXX=clang++
# tidy-ticket-perf-commit
# rustc-perf version from 2023-05-30
ENV PERF_COMMIT 8b2ac3042e1ff2c0074455a0a3618adef97156b1
# tidy-ticket-perf-commit
RUN curl -LS -o perf.zip https://github.com/rust-lang/rustc-perf/archive/$PERF_COMMIT.zip && \
    unzip perf.zip && \
    mv rustc-perf-$PERF_COMMIT rustc-perf && \
    rm perf.zip"#;
    let tag = "tidy-ticket-perf-commit";
    assert_eq!("76c8d9783e38e25a461355f82fcd7955", span_hash(source, tag, &mut true).unwrap());
}

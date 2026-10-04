#![crate_name = "foo"]

//! Reference to footnotes A[^1], B[^2] and C[^3].
//!
//! [^1]: Footnote A.
//! [^2]: Footnote B.
//! [^3]: Footnote C.

//@ has 'foo/index.html'
//@ has - '//*[@class="docblock"]/*[@class="footnotes"]/ol/li[@id="fn1"]/p' 'Footnote A'
//@ has - '//li[@id="fn1"]/p/a/@href' '#fnref1'
//@ has - '//*[@class="docblock"]/*[@class="footnotes"]/ol/li[@id="fn2"]/p' 'Footnote B'
//@ has - '//li[@id="fn2"]/p/a/@href' '#fnref2'
//@ has - '//*[@class="docblock"]/*[@class="footnotes"]/ol/li[@id="fn3"]/p' 'Footnote C'
//@ has - '//li[@id="fn3"]/p/a/@href' '#fnref3'

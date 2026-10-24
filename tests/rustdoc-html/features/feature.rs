//@ compile-flags: -Zunstable-options --feature-documentation x=tadam
//@ compile-flags: --feature-documentation 'y= yup '
//@ compile-flags: --feature-documentation 'z= another '
//@ compile-flags: --feature-documentation 'z-z=why not'

#![crate_name = "foo"]

// First we check they're correctly listed in the items list.
//@ has 'foo/index.html'
//@ count - '//dt/a[@class="feature"]' 4
//@ has - '//dt/a[@href="feature.x.html"]' 'x'
//@ has - '//dt/a[@href="feature.y.html"]' 'y'
//@ has - '//dt/a[@href="feature.z.html"]' 'z'
//@ has - '//dt/a[@href="feature.z-z.html"]' 'z-z'

// Then we check the "features" section is listed in the sidebar.
//@ has - '//*[@id="rustdoc-toc"]/ul/li/a[@href="#features"]' 'Features'

// And we check the files exist.
//@ has 'foo/feature.x.html' '//*[@class="docblock"]' 'tadam'
//@ has 'foo/feature.y.html' '//*[@class="docblock"]' 'yup'
//@ has 'foo/feature.z.html' '//*[@class="docblock"]' 'another'
//@ has 'foo/feature.z-z.html' '//*[@class="docblock"]' 'why not'

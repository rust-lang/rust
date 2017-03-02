- Start Date: 2014-11-01
- RFC PR: [#404](https://github.com/rust-lang/rfcs/pull/404)
- Rust Issue: [#18499](https://github.com/rust-lang/rust/issues/18499)

# Summary

When the compiler generates a dynamic library, alter the default behavior to
favor linking all dependencies statically rather than maximizing the number of
dynamic libraries. This behavior can be disabled with the existing
`-C prefer-dynamic` flag.

# Motivation

Long ago rustc used to only be able to generate dynamic libraries and as a
consequence all Rust libraries were distributed/used in a dynamic form. Over
time the compiler learned to create static libraries (dubbed rlibs). With this
ability the compiler had to grow the ability to choose between linking a library
either statically or dynamically depending on the available formats available to
the compiler.

Today's heuristics and algorithm are [documented in the compiler][linkage], and
the general idea is that as soon as "statically link all dependencies" fails
then the compiler maximizes the number of dynamic dependencies. Today there is
also not a method of instructing the compiler precisely what form intermediate
libraries should be linked in the source code itself. The linkage can be
"controlled" by passing `--extern` flags with only one per dependency where the
desired format is passed.

[linkage]: https://github.com/rust-lang/rust/blob/master/src/librustc/middle/dependency_format.rs

While functional, these heuristics do not allow expressing an important use case
of building a dynamic library as a final product (as opposed to an intermediate
Rust library) while having all dependencies statically linked to the final
dynamic library. This use case has been seen in the wild a number of times, and
the current workaround is to generate a `staticlib` and then invoke the linker
directly to convert that to a `dylib` (which relies on rustc generating PIC
objects by default).

The purpose of this RFC is to remedy this use case while largely retaining the
current abilities of the compiler today.

# Detailed design

In english, the compiler will change its heuristics for when a dynamic library
is being generated. When doing so, it will attempt to link all dependencies
statically, and failing that, will continue to maximize the number of dynamic
libraries which are linked in.

The compiler will also repurpose the `-C prefer-dynamic` flag to indicate that
this behavior is not desired, and the compiler should maximize dynamic
dependencies regardless.

In terms of code, the following patch will be applied to the compiler:

```patch
diff --git a/src/librustc/middle/dependency_format.rs b/src/librustc/middle/dependency_format.rs
index 8e2d4d0..dc248eb 100644
--- a/src/librustc/middle/dependency_format.rs
+++ b/src/librustc/middle/dependency_format.rs
@@ -123,6 +123,16 @@ fn calculate_type(sess: &session::Session,
             return Vec::new();
         }

+        // Generating a dylib without `-C prefer-dynamic` means that we're going
+        // to try to eagerly statically link all dependencies. This is normally
+        // done for end-product dylibs, not intermediate products.
+        config::CrateTypeDylib if !sess.opts.cg.prefer_dynamic => {
+            match attempt_static(sess) {
+                Some(v) => return v,
+                None => {}
+            }
+        }
+
         // Everything else falls through below
         config::CrateTypeExecutable | config::CrateTypeDylib => {},
     }
```

# Drawbacks

None currently, but the next section of alternatives lists a few other methods
of possibly achieving the same goal.

# Alternatives

## Disallow intermediate dynamic libraries

One possible solution to this problem is to completely disallow dynamic
libraries as a possible intermediate format for rust libraries. This would solve
the above problem in the sense that the compiler never has to make a choice.
This would also additionally cut the distribution size in roughly half because
only rlibs would be shipped, not dylibs.

Another point in favor of this approach is that the story for dynamic libraries
in Rust (for Rust) is also somewhat lacking with today's compiler. The ABI of a
library changes quite frequently for unrelated changes, and it is thus
infeasible to expect to ship a dynamic Rust library to later be updated
in-place without recompiling downstream consumers. By disallowing dynamic
libraries as intermediate formats in Rust, it is made quite obvious that a Rust
library cannot depend on another dynamic Rust library. This would be codifying
the convention today of "statically link all Rust code" in the compiler itself.

The major downside of this approach is that it would then be impossible to write
a plugin for Rust in Rust. For example compiler plugins would cease to work
because the standard library would be statically linked to both the `rustc`
executable as well as the plugin being loaded.

In the common case duplication of a library in the same process does not tend to
have adverse side effects, but some of the more flavorful features tend to
interact adversely with duplication such as:

* Globals with significant addresses (`static`s). These globals would all be
  duplicated and have different addresses depending on what library you're
  talking to.
* TLS/TLD. Any "thread local" or "task local" notion will be duplicated
  across each library in the process.

Today's design of the runtime in the standard library causes dynamically loaded
plugins with a statically linked standard library to fail very quickly as soon
as any runtime-related operations is performed. Note, however, that the runtime
of the standard library will likely be phased out soon, but this RFC considers
the cons listed above to be reasons to not take this course of action.

## Allow fine-grained control of linkage

Another possible alternative is to allow fine-grained control in the compiler to
explicitly specify how each library should be linked (as opposed to a blanked
prefer dynamic or not).

Recent forays with native libraries in Cargo has led to the conclusion that
hardcoding linkage into source code is often a hazard and a source of pain down
the line. The ultimate decision of how a library is linked is often not up to
the author, but rather the developer or builder of a library itself.

This leads to the conclusion that linkage control of this form should be
controlled through the command line instead, which is essentially already
possible today (via `--extern`). Cargo essentially does this, but the standard
libraries are shipped in dylib/rlib formats, causing the pain points listed in
the motivation.

As a result, this RFC does not recommend pursuing this alternative too far, but
rather considers the alteration above to the compiler's heuristics to be
satisfactory for now.

# Unresolved questions

None yet!

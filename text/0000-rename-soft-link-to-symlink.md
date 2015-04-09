- Feature Name: rename_soft_link_to_symlink
- Start Date: 2015-04-09
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Rename `std::fs::soft_link` to `std::fs::symlink` and provide a deprecated
`std::fs::soft_link` alias.

# Motivation

At some point in the split up version of rust-lang/rfcs#517,
`std::fs::symlink` was renamed to `sym_link` and then to `soft_link`.

The new name is somewhat surprising and can be difficult to find.  After a
poll of a number of different platforms and languages, every one appears to
contain `symlink`, `symbolic_link`, or some camel case variant of those for
their equivalent API.  Every piece of formal documentation found, for
both Windows and various Unix like platforms, used "symbolic link" exclusively
in prose.

Here are the names I found for this functionality on various platforms,
libraries, and languages:

* [POSIX/Single Unix Specification](http://pubs.opengroup.org/onlinepubs/009695399/functions/symlink.html): `symlink`
* [Windows](https://msdn.microsoft.com/en-us/library/windows/desktop/aa365680%28v=vs.85%29.aspx): `CreateSymbolicLink`
* [Objective-C/Swift](https://developer.apple.com/library/ios/documentation/Cocoa/Reference/Foundation/Classes/NSFileManager_Class/index.html#//apple_ref/occ/instm/NSFileManager/createSymbolicLinkAtPath:withDestinationPath:error:): `createSymbolicLinkAtPath:withDestinationPath:error:`
* [Java](https://docs.oracle.com/javase/7/docs/api/java/nio/file/Files.html): `createSymbolicLink`
* [C++ (Boost/draft standard)](http://en.cppreference.com/w/cpp/experimental/fs): `create_symlink`
* [Ruby](http://ruby-doc.org/core-2.2.0/File.html): `symlink`
* [Python](https://docs.python.org/2/library/os.html#os.symlink): `symlink`
* [Perl](http://perldoc.perl.org/functions/symlink.html): `symlink`
* [PHP](https://php.net/manual/en/function.symlink.php): `symlink`
* [Delphi](http://docwiki.embarcadero.com/Libraries/XE7/en/System.SysUtils.FileCreateSymLink): `FileCreateSymLink`
* PowerShell has no official version, but several community cmdlets ([one example](http://stackoverflow.com/questions/894430/powershell-hard-and-soft-links/894651#894651), [another example](https://gallery.technet.microsoft.com/scriptcenter/New-SymLink-60d2531e)) are named `New-SymLink`

The term "soft link", probably as a contrast with "hard link", is found
frequently in informal descriptions, but almost always in the form of a
parenthetical of an alternate phrase, such as "a symbolic link (or soft
link)".  I could not find it used in any formal documentation or APIs outside
of Rust.

The name `soft_link` was chosen to be shorter than `symbolic_link`, but
without using Unix specific jargon like `symlink`, to not give undue weight to
one platform over the other.  However, based on the evidence above it doesn't
have any precedent as a formal name for the concept or API.  Furthermore,
symbolic links themselves are a conept that were only relatively recently
(2007, with Windows Vista) introduced to Windows, with the [main motivator
being Unix compatibility](https://msdn.microsoft.com/en-us/library/windows/desktop/aa365680%28v=vs.85%29.aspx):

> Symbolic links are designed to aid in migration and application
> compatibility with UNIX operating systems. Microsoft has implemented its
> symbolic links to function just like UNIX links.

If you do a Google search for "[windows symbolic link](https://www.google.com/search?q=windows+symbolic+link&ie=utf-8&oe=utf-8)" or "[windows soft link](https://www.google.com/search?q=windows+soft+link&ie=utf-8&oe=utf-8)",
many of the documents you find start using "symlink" after introducing the
concept, so it seems to be a fairly common abbreviation for the full name even
among Windows developers and users.

# Detailed design

Rename `std::fs::soft_link` to `std::fs::symlink`, and provide a deprecated
`std::fs::soft_link` wrapper for backwards compatibility.  Update the
documentaiton to use "symbolic link" in prose, rather than "soft link".

# Drawbacks

This deprecates a stable API during the 1.0.0 beta, leaving an extra wrapper
around.

# Alternatives

Other choices for the name would be:

* The status quo, `soft_link`
* The original proposal from rust-lang/rfcs#517, `sym_link`
* The full name, `symbolic_link`

The first choice is non-obvious, for people coming from either Windows or
Unix.  It is a classic compromise, that makes everyone unhappy.

`sym_link` is slightly more consistent with the complementary `hard_link`
function, and treating "sym link" as two separate words has some precedent in
two of the Windows-targetted APIs, Delphi and some of the PowerShell cmdlets
observed.

The full name `symbolic_link`, is a bit long and cumbersome compared to most
of the rest of the API, but is explicit and is the term used in prose to
describe the concept everywhere, so shouldn't emphasize any one platform over
the other.

# Unresolved questions

If we deprecate `soft_link` now, early in the beta cycle, would it be
acceptable to remove it rather than deprecate it before 1.0.0, thus avoiding a
permanently stable but deprecated API right out the gate?

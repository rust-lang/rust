This directory holds method documentation that otherwise
would be duplicated across mutable and immutable pointers.

Note that most of the docs here are not the complete docs
for their corresponding method. This is for a few reasons:

1. Examples need to be different for mutable/immutable
   pointers, in order to actually call the correct method.
2. Link reference definitions are frequently different
   between mutable/immutable pointers, in order to link to
   the correct method.
   For example, `<*const T>::as_ref` links to
   `<*const T>::is_null`, while `<*mut T>::as_ref` links to
   `<*mut T>::is_null`.
3. Many methods on mutable pointers link to an alternate
   version that returns a mutable reference instead of
   a shared reference.

Always review the rendered docs manually when making
changes to these files to make sure you're not accidentally
splitting up a section.

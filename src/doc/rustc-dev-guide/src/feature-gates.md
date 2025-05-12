# Feature gates

This chapter is intended to provide basic help for adding, removing, and
modifying feature gates.

Note that this is specific to *language* feature gates; *library* feature gates use [a different
mechanism][libs-gate].

[libs-gate]: ./stability.md

## Adding a feature gate

See ["Stability in code"][adding] in the "Implementing new features" section for instructions.

[adding]: ./implementing_new_features.md#stability-in-code

## Removing a feature gate

[removing]: #removing-a-feature-gate

To remove a feature gate, follow these steps:

1. Remove the feature gate declaration in `rustc_feature/src/unstable.rs`.
   It will look like this:

   ```rust,ignore
   /// description of feature
   (unstable, $feature_name, "$version", Some($tracking_issue_number))
   ```

2. Add a modified version of the feature gate declaration that you just
   removed to `rustc_feature/src/removed.rs`:

   ```rust,ignore
   /// description of feature
   (removed, $old_feature_name, "$version", Some($tracking_issue_number),
    Some("$why_it_was_removed"))
   ```


## Renaming a feature gate

[renaming]: #renaming-a-feature-gate

To rename a feature gate, follow these steps (the first two are the same steps
to follow when [removing a feature gate][removing]):

1. Remove the old feature gate declaration in `rustc_feature/src/unstable.rs`.
   It will look like this:

   ```rust,ignore
   /// description of feature
   (unstable, $old_feature_name, "$version", Some($tracking_issue_number))
   ```

2. Add a modified version of the old feature gate declaration that you just
   removed to `rustc_feature/src/removed.rs`:

   ```rust,ignore
   /// description of feature
   /// Renamed to `$new_feature_name`
   (removed, $old_feature_name, "$version", Some($tracking_issue_number),
    Some("renamed to `$new_feature_name`"))
   ```

3. Add a feature gate declaration with the new name to
   `rustc_feature/src/unstable.rs`. It should look very similar to the old
   declaration:

   ```rust,ignore
   /// description of feature
   (unstable, $new_feature_name, "$version", Some($tracking_issue_number))
   ```


## Stabilizing a feature

See ["Updating the feature-gate listing"] in the "Stabilizing Features" chapter
for instructions. There are additional steps you will need to take beyond just
updating the declaration!


["Stability in code"]: ./implementing_new_features.md#stability-in-code
["Updating the feature-gate listing"]: ./stabilization_guide.md#updating-the-feature-gate-listing

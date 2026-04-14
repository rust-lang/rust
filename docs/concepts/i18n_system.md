# Internationalization (i18n) System

## Overview

Thing-OS includes a first-class i18n system that treats every UI-visible string as a translatable resource. This enables runtime locale switching without application restart.

## Architecture

### Core Types

#### `TextKey`
A stable identifier for translatable strings. Keys should be hierarchical and descriptive:
```rust
TextKey::new("ui.fonts.title")        // Font Explorer title
TextKey::new("ui.window.close")       // Close button
TextKey::new("ui.fonts.sample")       // Sample text
```

#### `LocaleId`
Identifies a locale (e.g., `en-US`, `la`, `syc`):
```rust
LocaleId::EN_US    // English (US)
LocaleId::LA       // Latin
LocaleId::SYC      // Syriac
```

#### `LocalizedText`
Combines a translation key with a fallback string:
```rust
const TITLE: LocalizedText = stem::t!("ui.fonts.title", "Fonts");

// Later, get the translated string:
let title_str = TITLE.get();  // Returns "Fonts" or "Litterae" depending on locale
```

#### `Catalog`
Maps text keys to translated strings for a specific locale:
```rust
const LATIN_CATALOG: &[(&str, &str)] = &[
    ("ui.fonts.title", "Litterae"),
    ("ui.fonts.explorer", "Explorator Litterarum"),
];
```

#### `Translator`
Global translation service that:
- Manages current locale state
- Provides translation lookups with fallback chain
- Tracks generation counter for UI cache invalidation

## Usage

### Defining Localized Text

Use the `t!` macro to define localized text constants:

```rust
use stem::i18n::LocalizedText;

const WINDOW_TITLE: LocalizedText = stem::t!("ui.window.title", "Window");
const BUTTON_LABEL: LocalizedText = stem::t!("ui.button.ok", "OK");
```

### Getting Translations

Call `.get()` to obtain the translated string:

```rust
let title = WINDOW_TITLE.get();  // Returns current locale's translation or fallback
```

### Runtime Locale Switching

Users can switch locales by pressing **F8** in Photosynthesis. This triggers:
1. `stem::i18n::cycle_locale()` - switches to next locale
2. Generation counter increment - signals UIs to redraw
3. UI applications detect the change and refresh with new translations

### Detecting Locale Changes

Applications can detect locale changes using the generation counter:

```rust
let mut last_i18n_gen = 0u64;

loop {
    let current_gen = stem::i18n::generation();
    if current_gen != last_i18n_gen {
        // Locale changed - redraw UI with new translations
        last_i18n_gen = current_gen;
        dirty = true;
    }
    
    // ... rest of render loop
}
```

## Translation Fallback Chain

When looking up a translation, the system follows this chain:
1. Current locale catalog
2. Default locale (en-US) catalog
3. Fallback string provided in `LocalizedText`

This ensures text is always displayed, even for untranslated strings.

## Adding New Locales

To add a new locale:

1. Define the locale constant in `stem/src/i18n/mod.rs`:
```rust
impl LocaleId {
    // ... existing locales
    pub const NEW_LOCALE: LocaleId = LocaleId("new-locale");
}
```

2. Create a translation catalog:
```rust
const NEW_LOCALE_CATALOG: &[(&str, &str)] = &[
    ("ui.fonts.title", "Translation"),
    // ... more translations
];
```

3. Register the catalog in `Translator::init()`:
```rust
catalogs.insert(LocaleId::NEW_LOCALE, Catalog::from_table(LocaleId::NEW_LOCALE, &NEW_LOCALE_CATALOG));
```

4. Update locale cycling logic in `set_locale()` and `cycle_locale()`

## Current Translation Coverage

### Latin Locale (`la`)
- Font Explorer UI strings (title, explorer, count labels)
- Sample text for font display
- Common window actions (close, minimize, maximize)

### Planned Locales
- **Syriac (`syc`)**: Placeholder defined, translations pending
- Additional locales can be added following the pattern above

## Example: Font Explorer

Font Explorer demonstrates i18n integration:

```rust
// Initialize i18n at startup
stem::i18n::init();

// Define localized text
const EXPLORER: LocalizedText = stem::t!("ui.fonts.explorer", "Font Explorer");
const COUNT_LABEL: LocalizedText = stem::t!("ui.fonts.count", "Fonts");
const SAMPLE: LocalizedText = stem::t!("ui.fonts.sample", "Sample text...");

// Use in UI
let header_text = alloc::format!("{} ({})", COUNT_LABEL.get(), fonts.len());
let scene = Scene::new().window(
    Window::new(win)
        .title(EXPLORER.get())  // Uses current locale
        .root(/* ... */)
);

// Detect locale changes
let current_i18n_gen = stem::i18n::generation();
if current_i18n_gen != last_i18n_gen {
    dirty = true;
    last_i18n_gen = current_i18n_gen;
}
```

## Testing

The i18n system includes comprehensive unit tests in `stem/src/i18n/mod.rs`:
- Text key creation and comparison
- Locale identifier validation
- Catalog lookups
- Translator locale switching
- Generation counter increments
- Translation fallback chain

Run tests with:
```bash
cargo test -p stem --lib i18n
```

## Design Principles

1. **First-Class**: i18n is built into the type system, not bolted on
2. **Zero Overhead**: Static catalogs, no filesystem or runtime overhead
3. **Type-Safe**: Keys are strings but wrapped in `TextKey` for clarity
4. **Always Valid**: Fallback chain ensures text is never missing
5. **Cache-Friendly**: Generation counter enables efficient UI invalidation
6. **No Strings in Code**: All UI text goes through the i18n system

## Future Enhancements

- [ ] Right-to-left (RTL) layout support for Syriac and other RTL scripts
- [ ] Plural forms handling (e.g., "1 font" vs "2 fonts")
- [ ] Date/time localization
- [ ] Number formatting per locale
- [ ] External translation file loading (when filesystem is available)
- [ ] Translation coverage reporting tools

# A Note on Accessibility in Rustdoc

`rustdoc` is WCAG-compliant[^1] and implements accessible documentation generation, with support for many accessibility practices and we strive to fully comply with WCAG Level AA[^2].

## Accessibility Practices Followed by Rustdoc

1. `rustdoc` only uses JavaScript to make your experience better, but `rustdoc` works perfectly well without it, meaning that all search indices and interactive features should work well without JavaScript. This ensures that no hidden scripts or interactive elements mess with screen readers and other accessibility tools.
2. `rustdoc` uses semantic HTML elements wherever possible.
    1. We use `<details>` instead of checkboxes so text can still be searched in hidden and collapsed content.
    2. We use the `title` attribute wherever applicable. The **`title`** global attribute contains text representing advisory information related to the element it belongs to.[^4]
3. All necessary elements are annotated by aria labels, to ensure that all elements have an accessible name[^4] for screen reader compatibility.
4. `rustdoc` has expansive support for keyboard navigation, such as keybinds for searching, focusing and interacting with the UI.

### WCAG 2.1 Compliance

1. `rustdoc`-generated documentation websites pass both level AA and AAA contrast guidelines.
2. `rustdoc` complies with the following sections for Level A and AA compliance:

| WCAG Criterion                                       | Evidence                                                                                                                         |
| ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **1.3.1 Info and Relationships (Level A)**           | Semantic HTML structure using headings, lists, tables, `<nav>`, and `<main>` preserves relationships for assistive technologies. |
| **1.3.2 Meaningful Sequence (Level A)**              | DOM order follows logical reading order; content remains meaningful when styles are removed.                                     |
| **1.3.3 Sensory Characteristics (Level A)**          | Instructions and content do not rely solely on shape, size, or visual location.                                                  |
| **1.4.4 Resize Text (Level AA)**                     | Content scales correctly with browser zoom; no fixed layouts prevent resizing up to 200%.                                        |
| **1.4.5 Images of Text (Level AA)**                  | Rustdoc does not render images of text; all text is real, selectable HTML text.                                                  |
| **1.4.10 Reflow (Level AA)**                         | Responsive layout reflows correctly at small viewport widths without requiring horizontal scrolling.                             |
| **1.4.12 Text Spacing (Level AA)**                   | Increasing line height, letter spacing, and word spacing via user CSS does not break layout or content.                          |
| **2.1.1 Keyboard (Level A)**                         | All primary functionality (navigation, search, settings) is operable using keyboard alone.                                       |
| **2.1.2 No Keyboard Trap (Level A)**                 | Focus can always move away from interactive elements using standard keyboard navigation.                                         |
| **2.1.4 Character Key Shortcuts (Level A)**          | Single-key shortcuts exist and can be disabled via rustdoc UI settings.                                                          |
| **2.2.1 Timing Adjustable (Level A)**                | No time-limited interactions or content that expires.                                                                            |
| **2.3.1 Three Flashes or Below Threshold (Level A)** | No flashing or strobing content present.                                                                                         |
| **2.4.1 Bypass Blocks (Level A)**                    | A mechanism is available to bypass blocks of content that are repeated on multiple web pages.                                    |
| **2.4.2 Page Titled (Level A)**                      | Each documentation page includes a unique and descriptive `<title>`.                                                             |
| **2.4.3 Focus Order (Level A)**                      | Keyboard focus order follows a logical and predictable sequence.                                                                 |
| **2.4.6 Headings and Labels (Level AA)**             | Clear, hierarchical headings and properly labeled controls are used throughout.                                                  |
| **3.1.1 Language of Page (Level A)**                 | HTML output sets the page language (`lang="en"` by default).                                                                     |
| **3.2.1 On Focus (Level A)**                         | Elements receiving focus do not trigger unexpected context changes.                                                              |
| **3.2.2 On Input (Level A)**                         | User input does not automatically cause navigation or major UI changes.                                                          |
| **3.2.3 Consistent Navigation (Level AA)**           | Navigation layout and placement remain consistent across pages.                                                                  |
| **3.2.4 Consistent Identification (Level AA)**       | UI components with the same function are consistently identified.                                                                |
| **3.3.1 Error Identification (Level A)**             | Errors (e.g., search issues) are communicated in text, not only visually.                                                        |
| **3.3.2 Labels or Instructions (Level A)**           | Form inputs such as search fields have accessible labels.                                                                        |
| **4.1.1 Parsing (Level A)**                          | Generated HTML is well-formed, with valid nesting and no duplicate IDs.                                                          |

## Known Accessibility Issues in Rustdoc

1. There are some contrast issues with certain parts of the webpage (see [#151422](https://github.com/rust-lang/rust/issues/151422))
2. WCAG 4.1.3 Status Messages is not implemented yet.

## Where to Get Help

If you experience any issues with accessibility while using `rustdoc`, please consider filling an issue at https://github.com/rust-lang/rust/issues.

[^1]: Consider reading https://www.w3.org/TR/WCAG22/ for more information about WCAG 2.2.

[^2]: Details about conformance levels can be found at https://www.w3.org/WAI/WCAG22/Understanding/conformance#levels.

[^3]: https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Global_attributes/title

[^4]: https://developer.mozilla.org/en-US/docs/Glossary/Accessible_name

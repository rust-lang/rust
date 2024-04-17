mat!(uni_literal, r"☃", "☃", Some((0, 3)));
mat!(uni_literal_plus, r"☃+", "☃", Some((0, 3)));
mat!(uni_literal_casei_plus, r"(?i)☃+", "☃", Some((0, 3)));
mat!(uni_class_plus, r"[☃Ⅰ]+", "☃", Some((0, 3)));
mat!(uni_one, r"\pN", "Ⅰ", Some((0, 3)));
mat!(uni_mixed, r"\pN+", "Ⅰ1Ⅱ2", Some((0, 8)));
mat!(uni_not, r"\PN+", "abⅠ", Some((0, 2)));
mat!(uni_not_class, r"[\PN]+", "abⅠ", Some((0, 2)));
mat!(uni_not_class_neg, r"[^\PN]+", "abⅠ", Some((2, 5)));
mat!(uni_case, r"(?i)Δ", "δ", Some((0, 2)));
mat!(uni_case_upper, r"\p{Lu}+", "ΛΘΓΔα", Some((0, 8)));
mat!(uni_case_upper_nocase_flag, r"(?i)\p{Lu}+", "ΛΘΓΔα", Some((0, 10)));
mat!(uni_case_upper_nocase, r"\p{L}+", "ΛΘΓΔα", Some((0, 10)));
mat!(uni_case_lower, r"\p{Ll}+", "ΛΘΓΔα", Some((8, 10)));

// Test the Unicode friendliness of Perl character classes.
mat!(uni_perl_w, r"\w+", "dδd", Some((0, 4)));
mat!(uni_perl_w_not, r"\w+", "⥡", None);
mat!(uni_perl_w_neg, r"\W+", "⥡", Some((0, 3)));
mat!(uni_perl_d, r"\d+", "1२३9", Some((0, 8)));
mat!(uni_perl_d_not, r"\d+", "Ⅱ", None);
mat!(uni_perl_d_neg, r"\D+", "Ⅱ", Some((0, 3)));
mat!(uni_perl_s, r"\s+", " ", Some((0, 3)));
mat!(uni_perl_s_not, r"\s+", "☃", None);
mat!(uni_perl_s_neg, r"\S+", "☃", Some((0, 3)));

// And do the same for word boundaries.
mat!(uni_boundary_none, r"\d\b", "6δ", None);
mat!(uni_boundary_ogham, r"\d\b", "6 ", Some((0, 1)));
mat!(uni_not_boundary_none, r"\d\B", "6δ", Some((0, 1)));
mat!(uni_not_boundary_ogham, r"\d\B", "6 ", None);

// Test general categories.
//
// We should test more, but there's a lot. Write a script to generate more of
// these tests.
mat!(uni_class_gencat_cased_letter, r"\p{Cased_Letter}", "Ａ", Some((0, 3)));
mat!(uni_class_gencat_cased_letter2, r"\p{gc=LC}", "Ａ", Some((0, 3)));
mat!(uni_class_gencat_cased_letter3, r"\p{LC}", "Ａ", Some((0, 3)));
mat!(
    uni_class_gencat_close_punctuation,
    r"\p{Close_Punctuation}",
    "❯",
    Some((0, 3))
);
mat!(
    uni_class_gencat_connector_punctuation,
    r"\p{Connector_Punctuation}",
    "⁀",
    Some((0, 3))
);
mat!(uni_class_gencat_control, r"\p{Control}", "\u{9f}", Some((0, 2)));
mat!(
    uni_class_gencat_currency_symbol,
    r"\p{Currency_Symbol}",
    "￡",
    Some((0, 3))
);
mat!(
    uni_class_gencat_dash_punctuation,
    r"\p{Dash_Punctuation}",
    "〰",
    Some((0, 3))
);
mat!(uni_class_gencat_decimal_numer, r"\p{Decimal_Number}", "𑓙", Some((0, 4)));
mat!(
    uni_class_gencat_enclosing_mark,
    r"\p{Enclosing_Mark}",
    "\u{A672}",
    Some((0, 3))
);
mat!(
    uni_class_gencat_final_punctuation,
    r"\p{Final_Punctuation}",
    "⸡",
    Some((0, 3))
);
mat!(uni_class_gencat_format, r"\p{Format}", "\u{E007F}", Some((0, 4)));
// See: https://github.com/rust-lang/regex/issues/719
mat!(uni_class_gencat_format_abbrev1, r"\p{cf}", "\u{E007F}", Some((0, 4)));
mat!(uni_class_gencat_format_abbrev2, r"\p{gc=cf}", "\u{E007F}", Some((0, 4)));
mat!(uni_class_gencat_format_abbrev3, r"\p{Sc}", "$", Some((0, 1)));
mat!(
    uni_class_gencat_initial_punctuation,
    r"\p{Initial_Punctuation}",
    "⸜",
    Some((0, 3))
);
mat!(uni_class_gencat_letter, r"\p{Letter}", "Έ", Some((0, 2)));
mat!(uni_class_gencat_letter_number, r"\p{Letter_Number}", "ↂ", Some((0, 3)));
mat!(
    uni_class_gencat_line_separator,
    r"\p{Line_Separator}",
    "\u{2028}",
    Some((0, 3))
);
mat!(
    uni_class_gencat_lowercase_letter,
    r"\p{Lowercase_Letter}",
    "ϛ",
    Some((0, 2))
);
mat!(uni_class_gencat_mark, r"\p{Mark}", "\u{E01EF}", Some((0, 4)));
mat!(uni_class_gencat_math, r"\p{Math}", "⋿", Some((0, 3)));
mat!(
    uni_class_gencat_modifier_letter,
    r"\p{Modifier_Letter}",
    "𖭃",
    Some((0, 4))
);
mat!(
    uni_class_gencat_modifier_symbol,
    r"\p{Modifier_Symbol}",
    "🏿",
    Some((0, 4))
);
mat!(
    uni_class_gencat_nonspacing_mark,
    r"\p{Nonspacing_Mark}",
    "\u{1E94A}",
    Some((0, 4))
);
mat!(uni_class_gencat_number, r"\p{Number}", "⓿", Some((0, 3)));
mat!(
    uni_class_gencat_open_punctuation,
    r"\p{Open_Punctuation}",
    "｟",
    Some((0, 3))
);
mat!(uni_class_gencat_other, r"\p{Other}", "\u{bc9}", Some((0, 3)));
mat!(uni_class_gencat_other_letter, r"\p{Other_Letter}", "ꓷ", Some((0, 3)));
mat!(uni_class_gencat_other_number, r"\p{Other_Number}", "㉏", Some((0, 3)));
mat!(
    uni_class_gencat_other_punctuation,
    r"\p{Other_Punctuation}",
    "𞥞",
    Some((0, 4))
);
mat!(uni_class_gencat_other_symbol, r"\p{Other_Symbol}", "⅌", Some((0, 3)));
mat!(
    uni_class_gencat_paragraph_separator,
    r"\p{Paragraph_Separator}",
    "\u{2029}",
    Some((0, 3))
);
mat!(
    uni_class_gencat_private_use,
    r"\p{Private_Use}",
    "\u{10FFFD}",
    Some((0, 4))
);
mat!(uni_class_gencat_punctuation, r"\p{Punctuation}", "𑁍", Some((0, 4)));
mat!(uni_class_gencat_separator, r"\p{Separator}", "\u{3000}", Some((0, 3)));
mat!(
    uni_class_gencat_space_separator,
    r"\p{Space_Separator}",
    "\u{205F}",
    Some((0, 3))
);
mat!(
    uni_class_gencat_spacing_mark,
    r"\p{Spacing_Mark}",
    "\u{16F7E}",
    Some((0, 4))
);
mat!(uni_class_gencat_symbol, r"\p{Symbol}", "⯈", Some((0, 3)));
mat!(
    uni_class_gencat_titlecase_letter,
    r"\p{Titlecase_Letter}",
    "ῼ",
    Some((0, 3))
);
mat!(
    uni_class_gencat_unassigned,
    r"\p{Unassigned}",
    "\u{10FFFF}",
    Some((0, 4))
);
mat!(
    uni_class_gencat_uppercase_letter,
    r"\p{Uppercase_Letter}",
    "Ꝋ",
    Some((0, 3))
);

// Test a smattering of properties.
mat!(uni_class_prop_emoji1, r"\p{Emoji}", "\u{23E9}", Some((0, 3)));
mat!(uni_class_prop_emoji2, r"\p{emoji}", "\u{1F21A}", Some((0, 4)));
mat!(
    uni_class_prop_picto1,
    r"\p{extendedpictographic}",
    "\u{1FA6E}",
    Some((0, 4))
);
mat!(
    uni_class_prop_picto2,
    r"\p{extendedpictographic}",
    "\u{1FFFD}",
    Some((0, 4))
);

// grapheme_cluster_break
mat!(
    uni_class_gcb_prepend,
    r"\p{grapheme_cluster_break=prepend}",
    "\u{11D46}",
    Some((0, 4))
);
mat!(
    uni_class_gcb_ri1,
    r"\p{gcb=regional_indicator}",
    "\u{1F1E6}",
    Some((0, 4))
);
mat!(uni_class_gcb_ri2, r"\p{gcb=ri}", "\u{1F1E7}", Some((0, 4)));
mat!(
    uni_class_gcb_ri3,
    r"\p{gcb=regionalindicator}",
    "\u{1F1FF}",
    Some((0, 4))
);
mat!(uni_class_gcb_lvt, r"\p{gcb=lvt}", "\u{C989}", Some((0, 3)));
mat!(uni_class_gcb_zwj, r"\p{gcb=zwj}", "\u{200D}", Some((0, 3)));

// word_break
mat!(uni_class_wb1, r"\p{word_break=Hebrew_Letter}", "\u{FB46}", Some((0, 3)));
mat!(uni_class_wb2, r"\p{wb=hebrewletter}", "\u{FB46}", Some((0, 3)));
mat!(uni_class_wb3, r"\p{wb=ExtendNumLet}", "\u{FF3F}", Some((0, 3)));
mat!(uni_class_wb4, r"\p{wb=WSegSpace}", "\u{3000}", Some((0, 3)));
mat!(uni_class_wb5, r"\p{wb=numeric}", "\u{1E950}", Some((0, 4)));

// sentence_break
mat!(uni_class_sb1, r"\p{sentence_break=Lower}", "\u{0469}", Some((0, 2)));
mat!(uni_class_sb2, r"\p{sb=lower}", "\u{0469}", Some((0, 2)));
mat!(uni_class_sb3, r"\p{sb=Close}", "\u{FF60}", Some((0, 3)));
mat!(uni_class_sb4, r"\p{sb=Close}", "\u{1F677}", Some((0, 4)));
mat!(uni_class_sb5, r"\p{sb=SContinue}", "\u{FF64}", Some((0, 3)));

// Test 'Vithkuqi' support, which was added in Unicode 14.
// See: https://github.com/rust-lang/regex/issues/877
mat!(
    uni_vithkuqi_literal_upper,
    r"(?i)^\u{10570}$",
    "\u{10570}",
    Some((0, 4))
);
mat!(
    uni_vithkuqi_literal_lower,
    r"(?i)^\u{10570}$",
    "\u{10597}",
    Some((0, 4))
);
mat!(uni_vithkuqi_word_upper, r"^\w$", "\u{10570}", Some((0, 4)));
mat!(uni_vithkuqi_word_lower, r"^\w$", "\u{10597}", Some((0, 4)));

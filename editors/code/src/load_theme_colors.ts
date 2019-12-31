import * as fs from 'fs';
import * as jsonc from 'jsonc-parser';
import * as path from 'path';
import * as vscode from 'vscode';

export interface TextMateRuleSettings {
    foreground?: string;
    background?: string;
    fontStyle?: string;
}

// Load all textmate scopes in the currently active theme
export function loadThemeColors(): Map<string, TextMateRuleSettings> {
    // Find out current color theme
    const themeName = vscode.workspace
        .getConfiguration('workbench')
        .get('colorTheme');

    if (typeof themeName !== 'string') {
        // console.warn('workbench.colorTheme is', themeName)
        return new Map();
    }
    return loadThemeNamed(themeName);
}

function loadThemeNamed(themeName: string): Map<string, TextMateRuleSettings> {
    function isTheme(extension: vscode.Extension<any>): boolean {
        return (
            extension.extensionKind === vscode.ExtensionKind.UI &&
            extension.packageJSON.contributes &&
            extension.packageJSON.contributes.themes
        );
    }

    let themePaths = vscode.extensions.all
        .filter(isTheme)
        .flatMap(ext => {
            return ext.packageJSON.contributes.themes
                .filter((it: any) => (it.id || it.label) === themeName)
                .map((it: any) => path.join(ext.extensionPath, it.path));
        })

    const res = new Map();
    for (const themePath of themePaths) {
        mergeInto(res, loadThemeFile(themePath))
    }

    const customizations: any = vscode.workspace.getConfiguration('editor').get('tokenColorCustomizations');
    mergeInto(res, loadColors(customizations?.textMateRules ?? []))

    return res;
}

function loadThemeFile(themePath: string): Map<string, TextMateRuleSettings> {
    let text;
    try {
        text = fs.readFileSync(themePath, 'utf8')
    } catch {
        return new Map();
    }
    const obj = jsonc.parse(text);
    const tokenColors = obj?.tokenColors ?? [];
    const res = loadColors(tokenColors);

    for (const include in obj?.include ?? []) {
        const includePath = path.join(path.dirname(themePath), include);
        const tmp = loadThemeFile(includePath);
        mergeInto(res, tmp);
    }

    return res;
}

interface TextMateRule {
    scope: string | string[];
    settings: TextMateRuleSettings;
}

function loadColors(textMateRules: TextMateRule[]): Map<string, TextMateRuleSettings> {
    const res = new Map();
    for (const rule of textMateRules) {
        const scopes = typeof rule.scope === 'string'
            ? [rule.scope]
            : rule.scope;
        for (const scope of scopes) {
            res.set(scope, rule.settings)
        }
    }
    return res
}

function mergeRuleSettings(
    defaultSetting: TextMateRuleSettings | undefined,
    override: TextMateRuleSettings,
): TextMateRuleSettings {
    return {
        foreground: defaultSetting?.foreground ?? override.foreground,
        background: defaultSetting?.background ?? override.background,
        fontStyle: defaultSetting?.fontStyle ?? override.fontStyle,
    }
}

function mergeInto(dst: Map<string, TextMateRuleSettings>, addition: Map<string, TextMateRuleSettings>) {
    addition.forEach((value, key) => {
        const merged = mergeRuleSettings(dst.get(key), value)
        dst.set(key, merged)
    })
}

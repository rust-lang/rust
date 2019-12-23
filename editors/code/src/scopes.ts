import * as fs from 'fs';
import * as jsonc from 'jsonc-parser';
import * as path from 'path';
import * as vscode from 'vscode';

export interface TextMateRule {
    scope: string | string[];
    settings: TextMateRuleSettings;
}

export interface TextMateRuleSettings {
    foreground: string | undefined;
    background: string | undefined;
    fontStyle: string | undefined;
}

// Current theme colors
const rules = new Map<string, TextMateRuleSettings>();

export function find(scope: string): TextMateRuleSettings | undefined {
    return rules.get(scope);
}

// Load all textmate scopes in the currently active theme
export function load() {
    // Remove any previous theme
    rules.clear();
    // Find out current color theme
    const themeName = vscode.workspace
        .getConfiguration('workbench')
        .get('colorTheme');

    if (typeof themeName !== 'string') {
        // console.warn('workbench.colorTheme is', themeName)
        return;
    }
    // Try to load colors from that theme
    try {
        loadThemeNamed(themeName);
    } catch (e) {
        // console.warn('failed to load theme', themeName, e)
    }
}

function filterThemeExtensions(extension: vscode.Extension<any>): boolean {
    return (
        extension.extensionKind === vscode.ExtensionKind.UI &&
        extension.packageJSON.contributes &&
        extension.packageJSON.contributes.themes
    );
}

// Find current theme on disk
function loadThemeNamed(themeName: string) {
    const themePaths = vscode.extensions.all
        .filter(filterThemeExtensions)
        .reduce((list, extension) => {
            return extension.packageJSON.contributes.themes
                .filter(
                    (element: any) =>
                        (element.id || element.label) === themeName,
                )
                .map((element: any) =>
                    path.join(extension.extensionPath, element.path),
                )
                .concat(list);
        }, Array<string>());

    themePaths.forEach(loadThemeFile);

    const tokenColorCustomizations: [any] = [
        vscode.workspace
            .getConfiguration('editor')
            .get('tokenColorCustomizations'),
    ];

    tokenColorCustomizations
        .filter(custom => custom && custom.textMateRules)
        .map(custom => custom.textMateRules)
        .forEach(loadColors);
}

function loadThemeFile(themePath: string) {
    const themeContent = [themePath]
        .filter(isFile)
        .map(readFileText)
        .map(parseJSON)
        .filter(theme => theme);

    themeContent
        .filter(theme => theme.tokenColors)
        .map(theme => theme.tokenColors)
        .forEach(loadColors);

    themeContent
        .filter(theme => theme.include)
        .map(theme => path.join(path.dirname(themePath), theme.include))
        .forEach(loadThemeFile);
}

function mergeRuleSettings(
    defaultSetting: TextMateRuleSettings | undefined,
    override: TextMateRuleSettings,
): TextMateRuleSettings {
    if (defaultSetting === undefined) {
        return override;
    }
    const mergedRule = defaultSetting;

    mergedRule.background = override.background || defaultSetting.background;
    mergedRule.foreground = override.foreground || defaultSetting.foreground;
    mergedRule.fontStyle = override.fontStyle || defaultSetting.foreground;

    return mergedRule;
}

function updateRules(
    scope: string,
    updatedSettings: TextMateRuleSettings,
): void {
    [rules.get(scope)]
        .map(settings => mergeRuleSettings(settings, updatedSettings))
        .forEach(settings => rules.set(scope, settings));
}

function loadColors(textMateRules: TextMateRule[]): void {
    textMateRules.forEach(rule => {
        if (typeof rule.scope === 'string') {
            updateRules(rule.scope, rule.settings);
        } else if (rule.scope instanceof Array) {
            rule.scope.forEach(scope => updateRules(scope, rule.settings));
        }
    });
}

function isFile(filePath: string): boolean {
    return [filePath].map(fs.statSync).every(stat => stat.isFile());
}

function readFileText(filePath: string): string {
    return fs.readFileSync(filePath, 'utf8');
}

function parseJSON(content: string): any {
    return jsonc.parse(content);
}

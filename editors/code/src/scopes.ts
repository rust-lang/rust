import * as fs from 'fs'
import * as jsonc from 'jsonc-parser'
import * as path from 'path'
import * as vscode from 'vscode'



export interface TextMateRule {
    scope: string | string[]
    settings: TextMateRuleSettings
}

export interface TextMateRuleSettings {
    foreground: string | undefined
    background: string | undefined
    fontStyle: string | undefined
}

// Current theme colors
const colors = new Map<string, TextMateRuleSettings>()

export function find(scope: string): TextMateRuleSettings | undefined {
    return colors.get(scope)
}

// Load all textmate scopes in the currently active theme
export function load() {
    // Remove any previous theme
    colors.clear()
    // Find out current color theme
    const themeName = vscode.workspace.getConfiguration('workbench').get('colorTheme')

    if (typeof themeName !== 'string') {
        console.warn('workbench.colorTheme is', themeName)
        return
    }
    // Try to load colors from that theme
    try {
        loadThemeNamed(themeName)
    } catch (e) {
        console.warn('failed to load theme', themeName, e)
    }
}

// Find current theme on disk
function loadThemeNamed(themeName: string) {
    for (const extension of vscode.extensions.all) {
        const extensionPath: string = extension.extensionPath
        const extensionPackageJsonPath: string = path.join(extensionPath, 'package.json')
        if (!checkFileExists(extensionPackageJsonPath)) {
            continue
        }
        const packageJsonText: string = readFileText(extensionPackageJsonPath)
        const packageJson: any = jsonc.parse(packageJsonText)
        if (packageJson.contributes && packageJson.contributes.themes) {
            for (const theme of packageJson.contributes.themes) {
                const id = theme.id || theme.label
                if (id === themeName) {
                    const themeRelativePath: string = theme.path
                    const themeFullPath: string = path.join(extensionPath, themeRelativePath)
                    loadThemeFile(themeFullPath)
                }
            }
        }

    }
    const customization: any = vscode.workspace.getConfiguration('editor').get('tokenColorCustomizations');
    if (customization && customization.textMateRules) {
        loadColors(customization.textMateRules)
    }
}

function loadThemeFile(themePath: string) {
    if (checkFileExists(themePath)) {
        const themeContentText: string = readFileText(themePath)
        const themeContent: any = jsonc.parse(themeContentText)

        if (themeContent && themeContent.tokenColors) {
            loadColors(themeContent.tokenColors)
            if (themeContent.include) {
                // parse included theme file
                const includedThemePath: string = path.join(path.dirname(themePath), themeContent.include)
                loadThemeFile(includedThemePath)
            }
        }
    }
}
function mergeRuleSettings(defaultRule: TextMateRuleSettings, override: TextMateRuleSettings): TextMateRuleSettings {
    const mergedRule = defaultRule;
    if (override.background) {
        mergedRule.background = override.background
    }
    if (override.foreground) {
        mergedRule.foreground = override.foreground
    }
    if (override.background) {
        mergedRule.fontStyle = override.fontStyle
    }
    return mergedRule;
}

function loadColors(textMateRules: TextMateRule[]): void {
    for (const rule of textMateRules) {

        if (typeof rule.scope === 'string') {
            const existingRule = colors.get(rule.scope);
            if (existingRule) {
                colors.set(rule.scope, mergeRuleSettings(existingRule, rule.settings))
            }
            else {
                colors.set(rule.scope, rule.settings)
            }
        } else if (rule.scope instanceof Array) {
            for (const scope of rule.scope) {
                const existingRule = colors.get(scope);
                if (existingRule) {
                    colors.set(scope, mergeRuleSettings(existingRule, rule.settings))
                }
                else {
                    colors.set(scope, rule.settings)
                }
            }
        }
    }
}

function checkFileExists(filePath: string): boolean {

    const stats = fs.statSync(filePath);
    if (stats && stats.isFile()) {
        return true;
    } else {
        console.warn('no such file', filePath)
        return false;
    }


}

function readFileText(filePath: string, encoding: string = 'utf8'): string {
    return fs.readFileSync(filePath, encoding);
}